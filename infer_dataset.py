import os
import sys
import cv2
import numpy as np
import torch
import argparse
import yaml
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import utils3d

from pathlib import Path

import colmap_interface

# Add project root and test directory to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'test'))

try:
    import infer as infer_module
    load_model = infer_module.load_model
    infer_raw = infer_module.infer_raw
    pred_to_vis = infer_module.pred_to_vis
except ImportError:
    # Fallback if test is treated as a package
    from test import infer as infer_module
    load_model = infer_module.load_model
    infer_raw = infer_module.infer_raw
    pred_to_vis = infer_module.pred_to_vis

def quat2mat(q):
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    return directions

def save_3d_points(points: np.array, colors: np.array, mask: np.array, filename: str):
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    mask = mask.reshape(-1)

    vertex_data = np.empty(mask.sum(), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = points[mask, 0]
    vertex_data['y'] = points[mask, 1]
    vertex_data['z'] = points[mask, 2]
    vertex_data['red'] = colors[mask, 0]
    vertex_data['green'] = colors[mask, 1]
    vertex_data['blue'] = colors[mask, 2]

    vertex_element = PlyElement.describe(vertex_data, 'vertex', comments=['point cloud'])
    PlyData([vertex_element], text=False).write(filename)

def compute_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute similarity transform (rotation, translation, uniform scale) using closed-form solution.
    Based on Umeyama's method.

    Args:
        src_pts: (N, 3) array of source points
        dst_pts: (N, 3) array of destination points

    Returns:
        transform: (4, 4) similarity transformation matrix
    """
    # Check for degenerate inputs
    if len(src_pts) < 3 or len(dst_pts) < 3:
        print(f"Warning: Not enough points for similarity transform (need >= 3, got {len(src_pts)})")
        return np.eye(4)

    # Compute centroids
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    # Center the point clouds
    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    # Compute scale
    src_scale = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    dst_scale = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))

    # Check for degenerate point clouds (all points at same location)
    if src_scale < 1e-10 and dst_scale < 1e-10:
        # Both point clouds are degenerate, just translate
        transform = np.eye(4)
        transform[:3, 3] = dst_centroid - src_centroid
        return transform
    elif src_scale < 1e-10:
        print(f"Warning: Source point cloud is degenerate (scale={src_scale})")
        scale = 1.0
    else:
        scale = dst_scale / src_scale

    # Normalize for rotation computation
    src_normalized = src_centered / (src_scale + 1e-10)
    dst_normalized = dst_centered / (dst_scale + 1e-10)

    # Compute rotation using SVD
    H = src_normalized.T @ dst_normalized

    # Check for NaN in H
    if not np.isfinite(H).all():
        print(f"Warning: H matrix contains NaN/Inf")
        return np.eye(4)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Verify R is a valid rotation matrix
    if not np.isfinite(R).all():
        print(f"Warning: Rotation matrix contains NaN/Inf")
        return np.eye(4)

    # Compute translation
    t = dst_centroid - scale * R @ src_centroid

    # Build 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t

    return transform

def apply_transform(src_pts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a similarity transform to a point cloud.
    The similarity transform maps src_pts to dst_pts: dst_pts = s*R*src_pts + t
    Args:
        src_pts: (N, 3) array of source points
        transform: (4, 4) similarity transformation matrix. Apply this transformation to src_pts to get dst_pts.
    Returns:
        dst_pts: (N, 3) array of destination points
    """
    if len(src_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Check for invalid input points
    if not np.isfinite(src_pts).all():
        print("Warning: Input points contain NaN/Inf values")
        valid_mask = np.isfinite(src_pts).all(axis=1)
        print(f"  {(~valid_mask).sum()} invalid points out of {len(src_pts)}")

    # Check for invalid transform
    if not np.isfinite(transform).all():
        print("Warning: Transform matrix contains NaN/Inf values")
        print(f"Transform:\n{transform}")
        raise ValueError("Invalid transform matrix")

    src_pts_homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    dst_pts_homo = (transform @ src_pts_homo.T).T
    dst_pts = dst_pts_homo[:, :3].copy()

    # Check output for issues
    if not np.isfinite(dst_pts).all():
        print("Warning: Transformed points contain NaN/Inf values")
        valid_mask = np.isfinite(dst_pts).all(axis=1)
        print(f"  {(~valid_mask).sum()} invalid points out of {len(dst_pts)}")

    return dst_pts


def compute_robust_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray, max_iterations: int = 1000, inlier_threshold: float = 0.1) -> np.ndarray:
    """
    Compute a robust similarity transform from source points to destination points.

    Uses RANSAC to robustly estimate rotation, translation, and uniform scale.
    The similarity transform maps src_pts to dst_pts: dst_pts = s*R*src_pts + t

    Args:
        src_pts: (N, 3) array of source points
        dst_pts: (N, 3) array of destination points
        max_iterations: maximum RANSAC iterations
        inlier_threshold: distance threshold for considering a point as an inlier

    Returns:
        transform: (4, 4) similarity transformation matrix. Apply this transformation to src_pts to get dst_pts.
    """
    assert len(src_pts) == len(dst_pts), "Point clouds must have same length"
    assert len(src_pts) >= 3, "Need at least 3 points to compute similarity transform"

    # Check for NaN/Inf in inputs
    if not np.isfinite(src_pts).all():
        print("Error: src_pts contains NaN/Inf values")
        return np.eye(4)
    if not np.isfinite(dst_pts).all():
        print("Error: dst_pts contains NaN/Inf values")
        return np.eye(4)

    N = len(src_pts)
    print(f"Computing robust similarity transform for {N} point pairs")
    best_inliers = 0
    best_transform = np.eye(4)

    # RANSAC loop
    for iteration in range(max_iterations):
        # Sample minimum number of points (3 for similarity transform)
        sample_indices = np.random.choice(N, size=3, replace=False)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]

        # Compute similarity transform from sample
        transform = compute_similarity_transform(src_sample, dst_sample)

        # Transform all source points
        transformed_pts = apply_transform(src_pts, transform)

        # Compute distances and count inliers
        distances = np.linalg.norm(transformed_pts - dst_pts, axis=1)
        inliers = distances < inlier_threshold
        num_inliers = np.sum(inliers)

        # Update best model if this is better
        if num_inliers > best_inliers:
            best_inliers = num_inliers

            # Refine using all inliers
            if num_inliers >= 3:
                best_transform = compute_similarity_transform(
                    src_pts[inliers],
                    dst_pts[inliers]
                )
            else:
                best_transform = transform

            # Early termination if we have very good fit
            if num_inliers > 0.98 * N:
                break

    print(f"RANSAC completed: best inliers = {best_inliers}/{N} ({100*best_inliers/N:.1f}%)")

    # Verify the final transform is valid
    if not np.isfinite(best_transform).all():
        print("Error: Final transform contains NaN/Inf")
        return np.eye(4)

    return best_transform



def process_dataset(dataset_folder: Path, config_path: str, vis_range="10m", cmap="Spectral"):
    # Paths
    out_folder = dataset_folder / "outfolder"

    colmap_folder = dataset_folder / "sparse"
    if not colmap_folder.exists():
        raise FileNotFoundError(f"Colmap folder not found: {colmap_folder}")

    # expecting keyframes folder for equirectangular images
    keyframes_folder = dataset_folder / "keyframes"
    if not keyframes_folder.exists():
        raise FileNotFoundError(f"Keyframes folder not found: {keyframes_folder}")

    depth_scale = 10 if vis_range == "10m" else 100

    ci = colmap_interface.ColmapInterface(dataset_folder, image_folder='images_cubemap')

    out_depth_vis = out_folder / "depth_vis"
    out_pts = out_folder / "pts"

    out_depth_vis.mkdir(parents=True, exist_ok=True)
    out_pts.mkdir(parents=True, exist_ok=True)

    # Load Config and Model
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model, device = load_model(config)

    infer_width = 1024

    for fid in tqdm(ci.frame_ids(), desc="Processing"):

        frame_info = ci.get_frame_info(fid)

        img_path = frame_info['keyframe_path']
        img_name = img_path.name.split('.')[0]

        sfm_depth_samples, _sample_ids, sfm_points = ci.spherical_frame_depth_samples(fid, infer_width)

        if not img_path.exists():
            print(f"⚠️ Image not found: {img_path}")
            continue

        # Inference
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️ Cannot read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize to 1024x512 for inference
        original_h, original_w = img_rgb.shape[:2]
        img_input = cv2.resize(img_rgb, (infer_width, int(infer_width/2)), interpolation=cv2.INTER_LINEAR)

        pred_depth = infer_raw(model, device, img_input) * depth_scale  # Scale depth

        # get pred depths for depth_samples
        uv_samples = np.array(sfm_depth_samples)[:, :2]
        picked_iu = uv_samples[..., 0].astype(int)
        picked_iv = uv_samples[..., 1].astype(int)
        pred_depth_samples = pred_depth[picked_iv, picked_iu]

        h, w = pred_depth.shape
        uv = utils3d.numpy.image_uv(width=w, height=h)
        dirs = spherical_uv_to_directions(uv)

        picked_dirs = dirs[picked_iv, picked_iu]
        picked_points = pred_depth_samples[..., None] * picked_dirs # [N, 3]

        transform = compute_robust_similarity_transform(picked_points, sfm_points)

        n_samples = len(sfm_points)
        save_3d_points(sfm_points, np.ones((n_samples, 3))*255, np.ones((n_samples, 1)).astype(np.bool), os.path.join(out_pts, f"{img_name}_sfm.ply"))
        save_3d_points(picked_points, np.ones((n_samples, 3))*255, np.ones((n_samples, 1)).astype(np.bool), os.path.join(out_pts, f"{img_name}_picked.ply"))


        # Back-project to camera frame
        points_cam = pred_depth[..., None] * dirs # [H, W, 3]

        points_cam_transformed = apply_transform(points_cam.reshape(-1, 3), transform)

        picked_transformed = apply_transform(picked_points.reshape(-1, 3), transform)
        save_3d_points(picked_transformed, np.ones((n_samples, 3))*255, np.ones((n_samples, 1)).astype(np.bool), os.path.join(out_pts, f"{img_name}_picked_transformed.ply"))

        n_samples = len(points_cam_transformed)
        save_3d_points(points_cam_transformed, img_input, np.ones((n_samples, 1)).astype(np.bool), os.path.join(out_pts, f"{img_name}_cam_transformed.ply"))



        # Visualization
        _, depth_color_rgb = pred_to_vis(pred_depth, vis_range=vis_range, cmap=cmap)

        # Save Colored Depth
        vis_path = os.path.join(out_depth_vis, f"{img_name}.png")
        cv2.imwrite(vis_path, cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR))

        # Generate Point Cloud
        # Use predicted depth (which is float32, likely normalized or in meters depending on model/vis)
        # Assuming pred_depth is metric or scaled consistently.
        # depth2point.py uses it directly.


        # Transform to world frame
        # P_world = R * P_cam + t
        R = frame_info['R']
        t = frame_info['t']

        # Apply transformation
        # points_cam is (H, W, 3). Reshape to (H*W, 3) for matrix multiplication
        points_flat = points_cam_transformed.reshape(-1, 3)
        points_world_flat = (points_flat - t) @ R
        points_world = points_world_flat.reshape(h, w, 3)

        # Save Point Cloud
        mask = pred_depth > 0
        ply_path = os.path.join(out_pts, f"{img_name}.ply")

        # We need colors for the points. Use the input image (resized)
        # img_input is 1024x512, matching pred_depth resolution
        save_3d_points(points_world, img_input, mask, ply_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=Path, help="Path to the dataset folder")
    parser.add_argument("--config", default="config/infer.yaml", help="Path to inference config")
    parser.add_argument("--vis", default="10m", choices=["100m", "10m"], help="Visualization range")
    parser.add_argument("--cmap", default="Spectral", help="Colormap")

    args = parser.parse_args()

    process_dataset(args.dataset, args.config, args.vis, args.cmap)
