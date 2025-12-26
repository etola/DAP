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

def process_dataset(dataset_folder, config_path, vis_range="10m", cmap="Spectral"):
    # Paths
    images_dir = os.path.join(dataset_folder, "images_orig")
    poses_file = os.path.join(dataset_folder, "tum_poses.txt")
    out_folder = os.path.join(dataset_folder, "outfolder")

    colmap_folder = os.path.join(dataset_folder, "sparse")
    if not os.path.exists(colmap_folder):
        raise FileNotFoundError(f"Colmap folder not found: {colmap_folder}")

    # expecting keyframes folder for equirectangular images
    keyframes_folder = os.path.join(dataset_folder, "keyframes")
    if not os.path.exists(keyframes_folder):
        raise FileNotFoundError(f"Keyframes folder not found: {keyframes_folder}")

    depth_scale = 10 if vis_range == "10m" else 100

    ci = colmap_interface.ColmapInterface(colmap_folder)

    out_depth_vis = os.path.join(out_folder, "depth_vis")
    out_pts = os.path.join(out_folder, "pts")

    os.makedirs(out_depth_vis, exist_ok=True)
    os.makedirs(out_pts, exist_ok=True)

    # Load Config and Model
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model, device = load_model(config)


    infer_width = 1024

    for fid in tqdm(ci.frame_ids(), desc="Processing"):

        frame_info = ci.frame_info(fid)

        img_name = frame_info['file_name'].basename()
        img_path = frame_info['file_path']

        depth_samples, _sample_ids = ci.spherical_frame_depth_samples(fid, infer_width)

        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        # Inference
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️ Cannot read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize to 1024x512 for inference
        original_h, original_w = img_rgb.shape[:2]
        img_input = cv2.resize(img_rgb, (infer_width, infer_width/2), interpolation=cv2.INTER_LINEAR)

        pred_depth = infer_raw(model, device, img_input) * depth_scale  # Scale depth

        # Visualization
        _, depth_color_rgb = pred_to_vis(pred_depth, vis_range=vis_range, cmap=cmap)

        # Save Colored Depth
        vis_path = os.path.join(out_depth_vis, f"{img_name}.png")
        cv2.imwrite(vis_path, cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR))

        # Generate Point Cloud
        # Use predicted depth (which is float32, likely normalized or in meters depending on model/vis)
        # Assuming pred_depth is metric or scaled consistently.
        # depth2point.py uses it directly.

        h, w = pred_depth.shape
        uv = utils3d.numpy.image_uv(width=w, height=h)
        dirs = spherical_uv_to_directions(uv)

        # Back-project to camera frame
        points_cam = pred_depth[..., None] * dirs # [H, W, 3]

        # Transform to world frame
        # P_world = R * P_cam + t
        R = frame_info['R']
        t = frame_info['t']

        # Apply transformation
        # points_cam is (H, W, 3). Reshape to (H*W, 3) for matrix multiplication
        points_flat = points_cam.reshape(-1, 3)
        points_world_flat = points_flat @ R.T + t
        points_world = points_world_flat.reshape(h, w, 3)

        # Save Point Cloud
        mask = pred_depth > 0
        ply_path = os.path.join(out_pts, f"{img_name}.ply")

        # We need colors for the points. Use the input image (resized)
        # img_input is 1024x512, matching pred_depth resolution
        save_3d_points(points_world, img_input, mask, ply_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str, help="Path to the dataset folder")
    parser.add_argument("--config", default="config/infer.yaml", help="Path to inference config")
    parser.add_argument("--vis", default="100m", choices=["100m", "10m"], help="Visualization range")
    parser.add_argument("--cmap", default="Spectral", help="Colormap")

    args = parser.parse_args()

    process_dataset(args.dataset_folder, args.config, args.vis, args.cmap)

