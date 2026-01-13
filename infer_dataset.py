import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import json
import pickle
from pathlib import Path

import colmap_interface
import open3d as o3d
from mvs_interface import SceneInterface
from parallel_executor import ParallelExecutor

def load_image(img_path: Path, width: int, height: int):
    if not img_path.exists():
        print(f"⚠️ Image not found: {img_path}")
        return None
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"⚠️ Cannot read image: {img_path}")
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return img_input

def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    return directions

def save_point_cloud(points: np.array, colors: np.array, filename: str):
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    vertex_data = np.empty(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    vertex_element = PlyElement.describe(vertex_data, 'vertex', comments=['point cloud'])
    PlyData([vertex_element], text=False).write(filename)

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

def compute_depth_scale_and_shift(src_depth_samples: np.ndarray, dst_depth_samples: np.ndarray) -> tuple[float, float]:
    """
    Compute scale and shift from source depth samples to destination depth samples. The transform maps src_depth_samples
    to dst_depth_samples: dst_depth_samples = scale * src_depth_samples + shift

    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
    Returns:
        scale: float
        shift: float
    """

    if len(src_depth_samples) != len(dst_depth_samples):
        raise ValueError("Source and destination depth samples must have the same length")
    if len(src_depth_samples) < 3 or len(dst_depth_samples) < 3:
        raise ValueError("Need at least 3 points to compute scale and shift transform")

    # Compute centroids
    src_centroid = np.mean(src_depth_samples)
    dst_centroid = np.mean(dst_depth_samples)

    # Center the depth samples
    src_centered = src_depth_samples - src_centroid
    dst_centered = dst_depth_samples - dst_centroid
    src_centered = src_centered.reshape(-1, 1)
    dst_centered = dst_centered.reshape(-1, 1)

    # Compute scale
    scale = np.sqrt(np.mean(dst_centered**2, axis=0)) / np.sqrt(np.mean(src_centered**2, axis=0))[0]

    # Compute translation
    shift = dst_centroid - scale * src_centroid

    return float(scale), float(shift)

def compute_robust_depth_scale_and_shift(src_depth_samples: np.ndarray, dst_depth_samples: np.ndarray, max_iterations: int = 1000, inlier_threshold: float = 0.05) -> tuple[float, float, int]:
    """
    Compute robust scale and shift from source depth samples to destination depth samples.
    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
        max_iterations: maximum RANSAC iterations
        inlier_threshold: threshold for considering a depth sample as an inlier in terms of abs percentage change wrt destination depth samples
    Returns:
        scale: float
        shift: float
        inliers: int
    """
    N = len(src_depth_samples)
    if N != len(dst_depth_samples):
        raise ValueError("Source and destination depth samples must have the same length")
    if N < 3:
        raise ValueError("Need at least 3 points to compute scale and shift transform")

    # Check for NaN/Inf in inputs
    if not np.isfinite(src_depth_samples).all():
        print("Error: src_depth_samples contains NaN/Inf values")
        return 1.0, 0.0

    best_inliers = 0
    best_scale = 1.0
    best_shift = 0.0

    inlier_set = []

    # RANSAC loop
    for iteration in range(max_iterations):

        if len(inlier_set) > 0.1*N and iteration < 100:
            sample_indices = np.random.choice(inlier_set, size=3, replace=False)
        else:
            sample_indices = np.random.choice(N, size=3, replace=False)

        src_sample = src_depth_samples[sample_indices]
        dst_sample = dst_depth_samples[sample_indices]

        # Compute scale and shift from sample
        scale, shift = compute_depth_scale_and_shift(src_sample, dst_sample)

        # Transform all source depth samples
        transformed_depth_samples = scale * src_depth_samples + shift

        # compute the error in terms of abs percentage change wrt destination depth samples
        error = np.abs(transformed_depth_samples - dst_depth_samples) / (dst_depth_samples + 1e-10)
        inliers = error < inlier_threshold
        num_inliers = np.sum(inliers)
        if num_inliers <= 3:
            continue

        # Update best model if this is better
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_scale, best_shift = scale, shift
            inlier_set = [ k for k in range(N) if inliers[k] ]

            if True: # Refine using all inliers
                best_scale, best_shift = compute_depth_scale_and_shift(src_depth_samples[inliers], dst_depth_samples[inliers])
                transformed_depth_samples = best_scale * src_depth_samples + best_shift
                error = np.abs(transformed_depth_samples - dst_depth_samples) / dst_depth_samples
                inliers = error < inlier_threshold
                num_inliers = np.sum(inliers)
                best_inliers = num_inliers
                inlier_set = [ k for k in range(N) if inliers[k] ]


        if best_inliers > 0.98 * N:
            break

    # print(f"RANSAC completed: best inliers = {best_inliers}/{N} ({100*best_inliers/N:.1f}%)")

    return best_scale, best_shift, best_inliers

def depth_map_to_cam_points(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert depth map to camera points.
    Args:
        depth_map: (H, W) depth map
    Returns:
        camera_points: (H, W, 3) array of camera points
    """
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uvd_full = np.dstack([u, v, depth_map]).reshape(-1, 3)
    points_cam = colmap_interface.spherical_uvd_to_ray_in_cam(uvd_full, (w, h))
    return points_cam.reshape(h, w, 3)

def depth_map_to_world_points(frame_info: dict, depth_map: np.ndarray) -> np.ndarray:
    """
    Convert depth map to world points.
    Args:
        frame_info: dict of the frame info
        depth_map: (H, W) depth map
    Returns:
        world_points: (H, W, 3) array of world points
    """
    cam_points = depth_map_to_cam_points(depth_map).reshape(-1, 3)
    R = frame_info['R']
    t = frame_info['t']
    h, w = depth_map.shape
    return ((cam_points - t) @ R).reshape(h, w, 3)

def world_points_to_uvd_in_spherical(world_points: np.ndarray, frame_info: dict, eq_width: int) -> np.ndarray:
    """
    Convert world points to uvd coordinates by projecting to the spherical image.
    Args:
        world_points: (N, 3) array of world points
        frame_info: dict of the frame info
    Returns:
        uvd: (N, 3) array of uvd coordinates
    """
    R = frame_info['R']
    t = frame_info['t']
    cam_points = (world_points @ R.T + t).reshape(-1, 3)
    depth = np.linalg.norm(cam_points, axis=1)
    uv = colmap_interface.spherical_img_from_cam((eq_width, eq_width//2), cam_points)
    uvd = np.dstack([uv[:, 0], uv[:, 1], depth]).reshape(-1, 3)
    return uvd

def register_with_sfm(ci: colmap_interface.ColmapInterface, fid: int, pred_depth: np.ndarray, dmin: float = 0.1, dmax: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Register the predicted depth map using the SfM points and return the world points and the depth map.
    Args:
        ci: colmap_interface.ColmapInterface
        fid: int frame id
        pred_depth: np.ndarray predicted depth map
        dmin: float minimum depth
        dmax: float maximum depth
    Returns:
        points_world: (H, W, 3) np.ndarray world points
        depth_map: (H, W) np.ndarray registered depth map
    """

    h, w = pred_depth.shape

    # get pred depths for depth_samples
    sfm_depth_samples, _sample_ids, sfm_points = ci.spherical_frame_depth_samples(fid, w)
    sfm_depth_samples = np.array(sfm_depth_samples)

    mask = (sfm_depth_samples[:, 2] >= dmin) & (sfm_depth_samples[:, 2] <= dmax)
    sfm_depth_samples = sfm_depth_samples[mask]

    if len(sfm_depth_samples) < 3:
        print(f"Not enough sfm depth samples for frame {fid}")
        return None, None

    uv_samples = sfm_depth_samples[:, :2]
    picked_iu = uv_samples[..., 0].astype(int)
    picked_iv = uv_samples[..., 1].astype(int)
    pred_depth_samples = pred_depth[picked_iv, picked_iu]

    scale, shift, inliers = compute_robust_depth_scale_and_shift(pred_depth_samples, sfm_depth_samples[:, 2], max_iterations=1000, inlier_threshold=0.05)
    if inliers < 30:
        print(f"Not enough inliers for frame {fid}: {inliers}")
        return None, None

    if abs(shift) > 0.5:
        print(f"Shift is too large for frame {fid}: {shift:.4f}")
        return None, None

    print(f"Scale/Shift/Inliers: {scale:.4f}, {shift:.4f}, {inliers:5d}")

    depth_map = scale * pred_depth + shift

    frame_info = ci.get_frame_info(fid)
    points_world = depth_map_to_world_points(frame_info, depth_map)

    return points_world, depth_map

def mask_out_edges(mask: np.ndarray, margin: int=3):
    """
    Mask out the edges of the mask by setting the edges to 0.
    Args:
        mask: (H, W) mask
        margin: margin width
    """
    mask[0:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, 0:margin] = 0
    mask[:, -margin:] = 0
    return mask

def mask_out_poles(mask: np.ndarray, margin: int=3):
    """
    Mask out the edges of the mask by setting the edges to 0.
    Args:
        mask: (H, W) mask
        margin: margin width
    """
    mask[0:margin, :] = 0
    mask[-margin:, :] = 0
    return mask

def filter_point_cloud(points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter the point cloud by the density of the points.
    Args:
        points: (N, 3) array of points
        colors: (N, 3) array of colors
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # filter out points that are too sparse
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=32, std_ratio=1.6)

    return np.array(pcd.points), np.array(pcd.colors)

def register_depthmaps_to_colmap(ci: colmap_interface.ColmapInterface, conf: dict):
    # # Paths
    # for fid in tqdm(ci.frame_ids(), desc="Processing"):
    #     _register_depthmap_to_colmap(ci, fid, conf)

    def _register_depthmap_to_colmap(fid: int, ci: colmap_interface.ColmapInterface, conf: dict) -> None:

        pred_dmap_folder = conf['pred_dmap_folder']
        dmap_folder = conf['dmap_folder']
        keyframe_mask_folder = conf['dataset'] / "keyframe_masks"
        out_pts = conf['out_pts']
        out_pts.mkdir(parents=True, exist_ok=True)
        
        frame_info = ci.get_frame_info(fid)
        img_path = frame_info['keyframe_path']
        img_name = img_path.name.split('.')[0]

        dmap_file = dmap_folder / f"{img_name}.npy"
        if dmap_file.exists():
            return

        if not img_path.exists():
            print(f"⚠️ Image not found: {img_path}")
            return

        dmap_file = pred_dmap_folder / f"{img_name}.npy"
        if not dmap_file.exists():
            print(f"⚠️ Depth map not found: {dmap_file}")
            return

        pred_depth = np.load(dmap_file)
        h, w = pred_depth.shape
        if w != 2*h:
            raise ValueError(f"Image width {w} should be twice the height {h} for equirectangulars image")

        img_mask_path = keyframe_mask_folder / f"{img_name}.png"
        if not img_mask_path.exists():
            img_mask = ci.get_spherical_mask_from_cubemap(fid, w, 3)
            cv2.imwrite(str(img_mask_path), img_mask)
        else:
            img_mask = cv2.imread(str(img_mask_path), cv2.IMREAD_GRAYSCALE)

        mask_out_poles(img_mask, margin=5)
        pred_depth[img_mask == 0] = 0

        points_world, depth_map = register_with_sfm(ci, fid, pred_depth, dmin=conf['dmin'], dmax=conf['dmax'])
        if points_world is None:
            print(f"No points world for frame {fid}")
            return

        validity_mask = (depth_map > conf['dmin']) & (depth_map < conf['dmax'])
        if validity_mask.sum() == 0:
            print(f"No valid points for frame {fid}")
            return

        depth_map[~validity_mask] = 0
        np.save(dmap_folder / f"{img_name}.npy", depth_map)

        if conf['save_registered_points']:
            img = load_image(frame_info['keyframe_path'], w, h)
            colors = img[validity_mask].reshape(-1, 3)
            points = points_world[validity_mask].reshape(-1, 3)
            points, colors = filter_point_cloud(points, colors)
            save_point_cloud(points, colors, out_pts / f"{img_name}-registered.ply")


    parallel_executor = ParallelExecutor(max_workers=4)
    parallel_executor.run_in_parallel_no_return(_register_depthmap_to_colmap, item_list=ci.frame_ids(), progress_desc="Registering depthmaps to colmap", ci=ci, conf=conf)


class Point:
    def __init__(self, X: np.ndarray, color: np.ndarray, fid: int, u: int, v: int):
        self.X = X.reshape(-1, 3)
        self.color = color
        self.visibility = [fid]
        self.u = u
        self.v = v

    def add_visibility(self, fid: int):
        self.visibility.append(fid)

class DepthData:

    def __init__(self, frame_info: dict):
        self.fid = frame_info['frame_id']
        self.img_path = frame_info['keyframe_path']
        self.dmap = None
        self.img = None
        self.merged_dmap = None
        self.valid_mask = None
        self.exported_mask = None
        self.neighbor_frame_ids = []
        self.valid_layers = {} # key: neighbor_fid, value: validity_mask for that neighbor frame

    def set(self, dmap: np.ndarray, neighbor_frame_ids: list[int]):
        self.dmap = dmap
        self.img = load_image(self.img_path, dmap.shape[1], dmap.shape[0])
        self.exported_mask = np.zeros((dmap.shape[0], dmap.shape[1]), dtype=bool)
        self.neighbor_frame_ids = neighbor_frame_ids
        self.valid_layers = {} # key: neighbor_fid, value: validity_mask for that neighbor frame

    def image_name(self) -> str:
        return self.img_path.name.split('.')[0]

    def h(self) -> int:
        return self.dmap.shape[0]

    def w(self) -> int:
        return self.dmap.shape[1]

    def save(self, folder: Path): # save depthdata structure
        with open(self.filename(folder), "wb") as f:
            pickle.dump(self, f)

    def load(self, folder: Path): # load depthdata structure
        with open(self.filename(folder), "rb") as f:
            return pickle.load(f)

    def filename(self, folder: Path = None) -> str:
        if folder is None:
            return f"{self.fid}-depthdata.pkl"
        else:
            return folder / f"{self.fid}-depthdata.pkl"



class ValidationManager:

    def __init__(self, ci: colmap_interface.ColmapInterface, conf: dict):

        self.ci = ci
        self.dmap_folder = conf['dmap_folder']
        self.out_pts = conf['out_pts']
        self.depthdata_folder = conf['out_folder'] / "depthdata"
        self.depthdata_folder.mkdir(parents=True, exist_ok=True)

        self.depth_data = {}

        self.depth_var_th = conf['depth_var_th']
        self.n_validity_th = conf['n_validity_th']
        self.n_neighbors = conf['n_neighbors']
        self.dmin = conf['dmin']
        self.dmax = conf['dmax']
        self.map_w = 0 # width of the equirectangular image

        self.face_id_map: np.ndarray = None # stores the face id for each pixel in the equirectangular image

    def init(self):
        for fid in tqdm(self.ci.frame_ids(), desc="Initializing depthdata"):
            frame_info = self.ci.get_frame_info(fid)
            dd = DepthData(frame_info)
            if dd.filename(self.depthdata_folder).exists():
                dd = dd.load(self.depthdata_folder)
                self.depth_data[fid] = dd
                continue
            else:
                dmap_path = self.dmap_folder / f"{dd.image_name()}.npy"
                if not dmap_path.exists():
                    print(f"dmap does not exist {fid}")
                    continue
                neighbor_frame_ids = self.ci.find_neighbor_frames(fid, min_points=10)
                if len(neighbor_frame_ids) < 2:
                    print(f"Not enough neighbors for frame {fid}")
                    continue
                dmap = np.load(dmap_path)
                dd.set(dmap, neighbor_frame_ids)
                dd.save(self.depthdata_folder)
                self.depth_data[fid] = dd

        for dd in self.depth_data.values():
            if dd.dmap is not None:
                self.map_w = dd.w()
                break

        if self.map_w == 0:
            raise ValueError("Width of the equirectangular image is not set")

        self.compute_frame_to_face_id_mapping()

    def compute_frame_to_face_id_mapping(self):

        """
        compute the mapping for the image_id visibility for the equirectangular images. ie, for each 
        u, v coordinate in the equirectangular image, find the image_id that can see the point.
        mapping is the same for all frames as we assume all the images are capture by the same rig of cameras.
        """

        self.face_id_map = np.zeros((self.map_w//2, self.map_w), dtype=int)

        # pick the first frame and compute the image_id_map
        frame_info = self.ci.get_frame_info(self.ci.frame_ids()[0])
        image_infos = frame_info['image_infos']

        u, v = np.meshgrid(np.arange(self.map_w//2), np.arange(self.map_w))
        uv = np.stack([u.flatten(), v.flatten()], axis=1)
        # get world points for the uv coordinates
        world_points = depth_map_to_world_points(frame_info, np.ones((self.map_w//2, self.map_w)))

        # project the world points to the image plane and get the image_id that can see the point
        for face_id in range(len(frame_info['image_ids'])):
            image_id = frame_info['image_ids'][face_id]
            image_info = image_infos[image_id]

            K = image_info['K']
            R = image_info['R']
            t = image_info['t']
            
            # project onto the image plane
            cam_points = (world_points.reshape(-1, 3) @ R.T + t).reshape(-1, 3)
            uv_proj = K @ cam_points.T
            depth = uv_proj[2, :]
            uv_proj = uv_proj[:2, :] / (depth[np.newaxis, :] + 1e-6)

            # check if the point is in the image and not behind the camera
            mask = (uv_proj[0, :] >= 0) & (uv_proj[0, :] < self.map_w) & (uv_proj[1, :] >= 0) & (uv_proj[1, :] < self.map_w//2) & (depth > 0)
            mask = mask.reshape(self.map_w//2, self.map_w)

            self.face_id_map[mask] = face_id

    def save_depthdata(self, fid: int):
        if fid in self.depth_data:
            self.depth_data[fid].save(self.depthdata_folder)

    def load_depthdata(self, fid: int):
        if fid in self.depth_data:
            return self.depth_data[fid].load(self.depthdata_folder)
        else:
            return None

    def validate_depthmap(self, fid: int) -> bool:

        if fid not in self.depth_data:
            print(f"fid {fid} not in depth data")
            return False

        fid_data = self.depth_data[fid]

        frame_info = self.ci.get_frame_info(fid)
        dmap_fid = fid_data.dmap

        mask_fid = (dmap_fid > self.dmin) & (dmap_fid < self.dmax)
        if mask_fid.sum() == 0:
            print(f"No valid points for frame {fid}")
            fid_data.valid_mask = mask_fid
            return False

        dmap_fid[~mask_fid] = 0

        merged_dmap = dmap_fid.copy()

        neighbor_frame_ids = fid_data.neighbor_frame_ids[:self.n_neighbors]

        validity_counts = np.zeros_like(dmap_fid, dtype=int)

        h, w = dmap_fid.shape

        for neighbor_fid in neighbor_frame_ids:
            if neighbor_fid not in self.depth_data:
                # print(f"neighbor {neighbor_fid} not in depth data")
                continue
            neighbor_depth_data = self.depth_data[neighbor_fid]
            neighbor_frame_info = self.ci.get_frame_info(neighbor_fid)
            neighbor_world_points = depth_map_to_world_points(neighbor_frame_info, neighbor_depth_data.dmap)

            # project the neighbor_world_points to the current frame and check the depth difference
            uvd = world_points_to_uvd_in_spherical(neighbor_world_points, frame_info, w)
            iu = uvd[:,0].astype(int)
            iv = uvd[:,1].astype(int)
            depth = uvd[:,2].astype(float)
            fid_depths = dmap_fid[iv, iu]
            depth_diff_percentage = np.abs(fid_depths - depth) / (depth + 1e-6)
            valid_depths = depth_diff_percentage < self.depth_var_th

            validity_layer = np.zeros_like(dmap_fid, dtype=bool)
            validity_layer[iv[valid_depths], iu[valid_depths]] = True

            merged_dmap[ iv[valid_depths], iu[valid_depths] ] += depth[valid_depths]

            fid_data.valid_layers[neighbor_fid] = validity_layer

            validity_counts[ validity_layer ] += 1

        validity_mask = validity_counts >= self.n_validity_th
        dmap_fid[~validity_mask] = 0

        merged_dmap[validity_mask] /= (validity_counts[validity_mask]+1)
        merged_dmap[~validity_mask] = 0

        fid_data.valid_mask = validity_mask
        fid_data.merged_dmap = merged_dmap

        return True

    def get_frame_cloud(self, fid: int):
        if fid not in self.depth_data:
            print(f"fid {fid} not in depth data")
            return None, None
        fid_data = self.depth_data[fid]
        if fid_data.merged_dmap is None:
            print(f"merged dmap is not available for frame {fid}")
            return None, None

        frame_info = self.ci.get_frame_info(fid)

        validity = fid_data.valid_mask
        dmap = fid_data.merged_dmap
        dmap[~validity] = 0

        h, w = dmap.shape
        points = depth_map_to_world_points(frame_info, dmap)
        img_input = fid_data.img
        colors = img_input[validity].reshape(-1, 3)
        points = points[validity].reshape(-1, 3)

        return points, colors

    def _export_point_cloud(self, fid: int):
        if fid not in self.depth_data:
            print(f"fid {fid} not in depth data")
            return
        fid_data = self.depth_data[fid]
        if fid_data.merged_dmap is None:
            print(f"merged dmap is not available for frame {fid}")
            return

        frame_info = self.ci.get_frame_info(fid)

        dmap = fid_data.merged_dmap
        validity_mask = fid_data.valid_mask
        exported_mask = fid_data.exported_mask

        # disable already exported points
        validity_mask[exported_mask] = False

        h, w = dmap.shape
        points = depth_map_to_world_points(frame_info, dmap)
        img_input = fid_data.img

        pixel_indices = np.argwhere(validity_mask)
        idx = pixel_indices[:, 0] * w + pixel_indices[:, 1]
        colors = img_input[validity_mask].reshape(-1, 3)
        points_valid = points[validity_mask].reshape(-1, 3)
        point_dict = {
            k: Point(p, c, fid, x, y)
            for k, p, c, x, y in zip(idx, points_valid, colors, pixel_indices[:, 1], pixel_indices[:, 0])
        }

        # extract visibility mask for each neighbor frame
        for nid,vlayer in fid_data.valid_layers.items():
            if nid not in self.depth_data:
                print(f"neighbor {nid} should be in depth data")
                continue
            vlayer = validity_mask & vlayer

            pixel_indices = np.argwhere(vlayer)
            indices = pixel_indices[:, 0] * w + pixel_indices[:, 1]

            face_ids = self.face_id_map[pixel_indices[:, 0], pixel_indices[:, 1]]

            nid_info = self.ci.get_frame_info(nid)

            for idx, face_id in zip(indices, face_ids):
                nid_image_id = nid_info['image_ids'][face_id]
                point_dict[idx].add_visibility(nid_image_id)

            points_vis = points[vlayer]
            # project the visible points to the neighbor frame & mark them as exported
            uvd = world_points_to_uvd_in_spherical(points_vis, self.ci.get_frame_info(nid), w)
            iu = uvd[:,0].astype(int)
            iv = uvd[:,1].astype(int)
            neighbor_depth_data = self.depth_data[nid]
            neighbor_depth_data.exported_mask[iv, iu] = True

        return point_dict

    def _init_scene_data(self) -> tuple[dict, dict]:
        """
        Initialize scene_data with platforms, images, and empty vertices.
        
        Returns:
            tuple of (scene_data, colmap_to_mvs_image mapping)
        """
        # Build platform and image mappings from colmap reconstruction
        camera_to_platform = {}
        platforms = []
        
        for camera_id, camera in sorted(self.ci.recon.cameras.items()):
            platform_idx = len(platforms)
            camera_to_platform[camera_id] = platform_idx
            
            K = camera.calibration_matrix().tolist()
            
            camera_entry = {
                "name": f"camera_{camera_id}",
                "bandName": "",
                "width": camera.width,
                "height": camera.height,
                "K": K,
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "C": [0.0, 0.0, 0.0],
            }
            
            platform = {
                "name": f"platform_{camera_id}",
                "cameras": [camera_entry],
                "poses": [],
            }
            platforms.append(platform)
        
        # Build image list and mapping from colmap image_id to mvs image index
        colmap_to_mvs_image = {}
        images = []
        platform_pose_counts = {}
        
        for image_id, image in sorted(self.ci.recon.images.items()):
            camera_id = image.camera_id
            platform_idx = camera_to_platform[camera_id]
            
            # Extract pose (rotation and camera center)
            pose = image.cam_from_world()
            R = pose.rotation.matrix()
            t = pose.translation
            C = (-R.T @ t).tolist()
            R = R.tolist()
            
            # Add pose to platform
            pose_entry = {"R": R, "C": C}
            if platform_idx not in platform_pose_counts:
                platform_pose_counts[platform_idx] = 0
            pose_idx = platform_pose_counts[platform_idx]
            platforms[platform_idx]["poses"].append(pose_entry)
            platform_pose_counts[platform_idx] += 1
            
            mvs_image_idx = len(images)
            colmap_to_mvs_image[image_id] = mvs_image_idx
            
            image_entry = {
                "name": image.name,
                "maskName": "",
                "platformID": platform_idx,
                "cameraID": 0,
                "poseID": pose_idx,
                "ID": mvs_image_idx,
                "minDepth": 0.0,
                "avgDepth": 0.0,
                "maxDepth": 0.0,
                "viewScores": [],
            }
            images.append(image_entry)
        
        # Initialize scene_data with empty vertices
        scene_data = {
            "platforms": platforms,
            "images": images,
            "vertices": [],
            "verticesNormal": [],
            "verticesColor": [],
            "lines": [],
            "linesNormal": [],
            "linesColor": [],
            "transform": [],
            "obb": {},
        }
        
        return scene_data, colmap_to_mvs_image

    def _export_frame_to_scene(self, fid: int, scene_data: dict, colmap_to_mvs_image: dict) -> int:
        """
        Export a single frame's points directly into scene_data vertices.
        
        Args:
            fid: frame id to export
            scene_data: scene_data dictionary to append vertices to
            colmap_to_mvs_image: mapping from colmap image_id to mvs image index
            
        Returns:
            number of points exported
        """
        if fid not in self.depth_data:
            return 0
        fid_data = self.depth_data[fid]
        if fid_data.merged_dmap is None:
            return 0

        frame_info = self.ci.get_frame_info(fid)

        dmap = fid_data.merged_dmap
        validity_mask = fid_data.valid_mask.copy()
        exported_mask = fid_data.exported_mask

        # disable already exported points
        validity_mask[exported_mask] = False

        h, w = dmap.shape
        points = depth_map_to_world_points(frame_info, dmap)
        img_input = fid_data.img

        pixel_indices = np.argwhere(validity_mask)
        if len(pixel_indices) == 0:
            return 0
            
        idx_array = pixel_indices[:, 0] * w + pixel_indices[:, 1]
        colors = img_input[validity_mask].reshape(-1, 3)
        points_valid = points[validity_mask].reshape(-1, 3)
        
        # Build visibility for each point: start with empty lists
        # Each point starts with visibility from current frame's images
        n_points = len(idx_array)
        point_visibility = [[] for _ in range(n_points)]
        
        # Map from idx to position in our arrays
        idx_to_pos = {idx: pos for pos, idx in enumerate(idx_array)}

        # Extract visibility mask for each neighbor frame
        for nid, vlayer in fid_data.valid_layers.items():
            if nid not in self.depth_data:
                continue
            vlayer = validity_mask & vlayer

            vis_pixel_indices = np.argwhere(vlayer)
            if len(vis_pixel_indices) == 0:
                continue
                
            vis_indices = vis_pixel_indices[:, 0] * w + vis_pixel_indices[:, 1]
            face_ids = self.face_id_map[vis_pixel_indices[:, 0], vis_pixel_indices[:, 1]]

            nid_info = self.ci.get_frame_info(nid)

            for idx, face_id in zip(vis_indices, face_ids):
                if idx in idx_to_pos:
                    nid_image_id = nid_info['image_ids'][face_id]
                    if nid_image_id in colmap_to_mvs_image:
                        mvs_img_id = colmap_to_mvs_image[nid_image_id]
                        point_visibility[idx_to_pos[idx]].append(mvs_img_id)

            points_vis = points[vlayer]
            # project the visible points to the neighbor frame & mark them as exported
            uvd = world_points_to_uvd_in_spherical(points_vis, self.ci.get_frame_info(nid), w)
            iu = uvd[:, 0].astype(int)
            iv = uvd[:, 1].astype(int)
            neighbor_depth_data = self.depth_data[nid]
            neighbor_depth_data.exported_mask[iv, iu] = True

        # Append vertices directly to scene_data
        vertices = scene_data["vertices"]
        vertices_color = scene_data["verticesColor"]
        
        for i in range(n_points):
            xyz = points_valid[i].tolist()
            
            # Build views from visibility (remove duplicates)
            seen_mvs_ids = set(point_visibility[i])
            views = [{"imageID": mvs_id, "confidence": 1.0} for mvs_id in seen_mvs_ids]
            
            vertex = {"X": xyz, "views": views}
            vertices.append(vertex)
            
            # Convert RGB to BGR for MVS format
            rgb = colors[i]
            color = {"c": [int(rgb[2]), int(rgb[1]), int(rgb[0])]}
            vertices_color.append(color)
        
        return n_points

    def export_scene_data(self) -> dict:
        """
        Export all depth data directly to scene_data format.
        
        This is memory-efficient as it builds vertices directly into scene_data
        without intermediate point_dict storage.
        
        Returns:
            dict: scene_data dictionary compatible with SceneInterface.save()
        """
        # Initialize scene_data with platforms and images
        scene_data, colmap_to_mvs_image = self._init_scene_data()
        
        # Export each frame's points directly to scene_data
        total_points = 0
        for fid in tqdm(self.depth_data, desc="Exporting to scene data"):
            n_points = self._export_frame_to_scene(fid, scene_data, colmap_to_mvs_image)
            total_points += n_points
        
        print(f"Exported {total_points} points to scene data")
        return scene_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=Path, help="Path to the dataset folder")
    parser.add_argument("--output", "-o", type=Path, default="outfolder", help="Path to the output folder relative to dataset. default [outfolder]")
    parser.add_argument("--colmap_folder", "-c", type=Path, default="sparse", help="Path to the colmap folder relative to dataset default [sparse]")
    parser.add_argument("--dmin", type=float, default=0.3, help="Minimum depth, default [0.3]")
    parser.add_argument("--dmax", type=float, default=7.0, help="Maximum depth, default [7.0]")
    parser.add_argument("--depth_var_th", type=float, default=0.04, help="Depth variance threshold, default [0.04]")
    parser.add_argument("--n_validity_th", type=int, default=3, help="Minimum number of valid points, default [3]")
    parser.add_argument("--n_neighbors", type=int, default=10, help="Number of neighbors, default [10]")
    parser.add_argument("--save_registered_points", "-r", action="store_true", help="Save registered points (default: False)")
    parser.add_argument("--save_validated_points", "-v", action="store_true", help="Save validated points (default: False)")

    args = parser.parse_args()

    conf = {
        "dataset": args.dataset,
        "colmap_folder": args.dataset / args.colmap_folder,
        "out_folder": args.dataset / args.output,
        "dmap_folder": args.dataset / args.output / "dmaps",
        "pred_dmap_folder": args.dataset / args.output / "pred_dmaps",
        "out_pts": args.dataset / args.output / "pts",
        "dmin": args.dmin,
        "dmax": args.dmax,
        "save_registered_points": args.save_registered_points,
        "depth_var_th": args.depth_var_th,
        "n_validity_th": args.n_validity_th,
        "n_neighbors": args.n_neighbors,
        "save_validated_points": args.save_validated_points,
    }
    if not conf['pred_dmap_folder'].exists():
        raise FileNotFoundError(f"Predicted depth map folder not found: {conf['pred_dmap_folder']}")

    conf['dmap_folder'].mkdir(parents=True, exist_ok=True)
    conf['out_pts'].mkdir(parents=True, exist_ok=True)

    if not conf["colmap_folder"].exists():
        raise FileNotFoundError(f"Colmap folder not found: {conf['colmap_folder']}")

    ci = colmap_interface.ColmapInterface(conf["dataset"], model_path=conf["colmap_folder"].relative_to(conf["dataset"]), image_folder='images_cubemap')
    ci.save_point_cloud(conf['out_folder'] / "colmap_points.ply")

    register_depthmaps_to_colmap(ci, conf)
    frame_ids = ci.frame_ids()

    validation_manager = ValidationManager(ci, conf)
    validation_manager.init()

    for fid in tqdm(frame_ids, desc="Validating depthmaps"):
        if not validation_manager.validate_depthmap(fid):
            continue
        validation_manager.save_depthdata(fid)
        points, colors = validation_manager.get_frame_cloud(fid)
        if points is None or colors is None:
            continue

        if conf['save_validated_points']:
            img_name = validation_manager.depth_data[fid].image_name()
            save_point_cloud(points, colors, conf['out_pts'] / f"{img_name}-validated.ply")

    # Export directly to scene_data (memory-efficient)
    scene_data = validation_manager.export_scene_data()
    SceneInterface.save(conf['out_folder'] / "scene_dense.mvs", scene_data)

