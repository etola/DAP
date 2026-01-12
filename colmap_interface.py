import argparse
import pycolmap
import numpy as np
import os
import cv2
from pathlib import Path
from parallel_executor import ParallelExecutor

def compute_relative_pose(R1, t1, R2, t2):
    """
    Computes the relative pose (R_rel, t_rel) from camera 1 to camera 2.
    Args:
        R1: Rotation matrix of camera 1 (3x3)
        t1: Translation vector of camera 1 (3,)
        R2: Rotation matrix of camera 2 (3x3)
        t2: Translation vector of camera 2 (3,)
    Returns:
        R_rel: Relative rotation matrix (3x3)
        t_rel: Relative translation vector (3,)
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel

def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    """Project rays into a 360 panorama (spherical) image. (copied from colmap)
    Args:
        image_size: (width, height) of the equirectangular image
        rays_in_cam: (N, 3) array of 3D ray directions in camera coordinates
    Returns:
        uv: (N, 2) array of pixel coordinates in the equirectangular image
    """
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size

def spherical_uvd_to_ray_in_cam(uvd: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Convert equirectangular pixel coordinates and depth to 3D rays in camera coordinates.
    Args:
        uvd: (N, 3) array of (u, v, depth) in the equirectangular image
        image_size: (width, height) of the equirectangular image
    Returns:
        rays_in_cam: (N, 3) array of 3D ray directions in camera coordinates
    """
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if uvd.ndim != 2 or uvd.shape[1] != 3:
        raise ValueError(f"{uvd.shape=} but expected (N,3).")
    u = uvd[:, 0] / image_size[0]
    v = uvd[:, 1] / image_size[1]
    depth = uvd[:, 2]
    yaw = (u * 2 - 1) * np.pi
    pitch = (1 - 2*v) * np.pi / 2
    x = depth * np.sin(yaw) * np.cos(pitch)
    y = -depth * np.sin(pitch)
    z = depth * np.cos(yaw) * np.cos(pitch)
    rays_in_cam = np.stack([x, y, z], -1)
    return rays_in_cam

class ColmapInterface:

    def __init__(self, workfolder: Path, model_path: str = "sparse", image_folder: str = "images", n_min_tracks: int = 3):

        self.workfolder = workfolder.resolve()
        self.model_path = self.workfolder / model_path
        self.image_folder = self.workfolder / image_folder
        self.pano_face_mask_folder = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        if not self.image_folder.exists():
            print(f"Warning: Image folder {image_folder} does not exist")
            raise FileNotFoundError(f"Image folder {image_folder} does not exist")

        # panaroma case:
        if image_folder == "images_cubemap":
            self.pano_face_mask_folder = self.workfolder / "binary_masks"
            if not self.pano_face_mask_folder.exists():
                self.pano_face_mask_folder = None

        self.recon = pycolmap.Reconstruction(self.model_path)
        self.n_min_tracks = n_min_tracks

        # active point ids with minimum number of tracks
        self.active_point_ids = [
            pid for pid, p3d in self.recon.points3D.items()
            if p3d.track.length() >= self.n_min_tracks
        ]

        print(f"Active point number (vis={self.n_min_tracks}): {len(self.active_point_ids)}")

        self.scene_bbox = self.compute_scene_bounding_box()

        print(f"Scene bounding box: {self.scene_bbox}")

        # stores the cached point ids per image_id
        self._image_point_ids = {}

    def _cache_image_point_ids(self):
        # Pre-compute image_id sets for each active point to avoid repeated lookups
        point_image_sets = {}
        for pid in self.active_point_ids:
            p3d = self.recon.points3D[pid]
            # p3d.track.elements is a list of TrackElements, each has .image_id
            point_image_sets[pid] = {el.image_id for el in p3d.track.elements}

        # cache point ids per image_id to have faster access (use the active point ids)
        self._image_point_ids = {}
        for image_id in self.recon.images.keys():
            self._image_point_ids[image_id] = [
                pid for pid in self.active_point_ids
                if image_id in point_image_sets[pid]
            ]

    def ensure_image_point_ids_cached(self):
        if not self._image_point_ids:
            self._cache_image_point_ids()

    def frame_ids(self) -> list[int]:
        """
        Returns the frame ids sorted by the frame id.
        """
        return sorted(list(self.recon.frames.keys()))

    def image_ids(self) -> list[int]:
        return list(self.recon.images.keys())

    def get_image_point_ids(self, image_id):
        self.ensure_image_point_ids_cached()
        return self._image_point_ids[image_id]

    def get_active_points(self):
        return [self.recon.points3D[pid] for pid in self.active_point_ids]

    def get_active_points_xyz(self):
        return [p3d.xyz for p3d in self.get_active_points()]

    def get_active_points_rgb(self):
        return [p3d.color for p3d in self.get_active_points()]

    def get_info(self):
        return {
            "num_images": self.recon.num_images(),
            "num_frames": self.recon.num_frames(),
            "num_cameras": self.recon.num_cameras(),
            "num_rigs": self.recon.num_rigs(),
            "num_points": self.recon.num_points3D()
        }

    def get_image_info(self, image_id):
        if image_id not in self.recon.images:
            raise ValueError(f"Image ID {image_id} not found")

        image = self.recon.images[image_id]
        camera = self.recon.cameras[image.camera_id]

        # K matrix
        K = camera.calibration_matrix()

        # Distortion params
        dist_params = camera.params[camera.extra_params_idxs()]

        # Extrinsics
        # cam_from_world() returns Rigid3d
        pose = image.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation
        C = -R.T @ t

        # try possible image paths:
        img_path = self.image_folder / image.name
        if not img_path.exists():
            img_path = self.image_folder / "images" / image.name
        if not img_path.exists():
            img_path = self.image_folder / "images" / image.name.basename()
        if not img_path.exists():
            img_path = self.image_folder / "images_cubemap" / image.name
        if not img_path.exists():
            print(f"Image {image_id} not found: {image.name}")
            # raise FileNotFoundError(f"Image {image_id} not found: {image.name}")

        return {
            "image_id": image_id,
            "image_name": image.name,
            "image_path": img_path,
            "camera_id": image.camera_id,
            "width": camera.width,
            "height": camera.height,
            "K": K,
            "distortion_params": dist_params,
            "R": R,
            "t": t,
            "c": C,
            "point_ids": self.get_image_point_ids(image_id)
        }

    def get_point_info(self, point_id) -> dict:
        if point_id not in self.recon.points3D:
            raise ValueError(f"Point ID {point_id} not found")
        p3d = self.recon.points3D[point_id]

        return {
            "point_id": point_id,
            "xyz": p3d.xyz,
            "color": p3d.color,
            "image_ids": [el.image_id for el in p3d.track.elements]
        }

    def get_frame_info(self, frame_id):
        """
        Returns the frame information for a given frame id.
        Args:
            frame_id: ID of the frame to process
        Returns:
            dict: frame information containing:
                - keyframe_path: path to the keyframe image
                - rig_id: ID of the rig
                - frame_id: ID of the frame
                - image_ids: list of image IDs in the frame
                - point_ids: list of point IDs in the frame
                - image_infos: dictionary of image information
                - sensor_mapping: dictionary of sensor mapping
                - R: rotation matrix from world to camera coordinates
                - t: translation vector from world to camera coordinates
                - c: camera center in world coordinates
        """
        if frame_id not in self.recon.frames:
            raise ValueError(f"Frame ID {frame_id} not found")

        frame = self.recon.frames[frame_id]

        # Extract image IDs from frame data
        frame_images = [(data.id, data.sensor_id.id) for data in frame.data_ids]

        image_ids = []
        sensor_mapping = {}
        image_infos = {}
        point_ids = []

        for img_id, sensor_id in frame_images:
            image_ids.append(img_id)
            try:
                info = self.get_image_info(img_id)
                info.update({"sensor_id": sensor_id})
                image_infos[img_id] = info
                point_ids.extend(info['point_ids'])
                sensor_mapping[sensor_id] = img_id
            except ValueError:
                print(f"Warning: Image {img_id} in Frame {frame_id} not found in images")

        keyframe_path = None
        if len(image_infos) == 6: # assuming cubemap images
            first_image_name = image_infos[image_ids[5]]['image_name']
            keyframe_name = first_image_name.split('/')[1]
            keyframe_path = self.workfolder / "keyframes" / keyframe_name

        rig_from_world: pycolmap.Rigid3d = frame.rig_from_world
        R = rig_from_world.rotation.matrix()
        t = rig_from_world.translation

        return {
            "keyframe_path": keyframe_path,
            "rig_id": frame.rig_id,
            "frame_id": frame_id,
            "image_ids": image_ids,
            "point_ids": point_ids,
            "image_infos": image_infos,
            "sensor_mapping": sensor_mapping, # stores the order of images in the rig
            "R" : R, # rotation matrix from world to camera coordinates
            "t" : t, # translation vector from world to camera coordinates
            "c" : -R.T @ t
        }

    def get_joint_points(self, frame_id1: int, frame_id2: int) -> list[int]:
        """
        Returns the joint point ids of the two frames.
        Args:
            frame_id1: ID of the first frame
            frame_id2: ID of the second frame
        Returns:
            list of joint point ids
        """
        frame_info1 = self.get_frame_info(frame_id1)
        frame_info2 = self.get_frame_info(frame_id2)
        point_ids1 = frame_info1['point_ids']
        point_ids2 = frame_info2['point_ids']
        return list(set(point_ids1) & set(point_ids2))

    def get_joint_points_any(self, frame_id1: int, frame_ids: list[int]) -> list[int]:
        """
        Returns the joint points of a single frame with the other frames ie if a point is in frame_id1 and any of the other frames, it is returned.
        Args:
            frame_id1: ID of the frame to process
            frame_ids: list of other frame IDs to compare with
        Returns:
            list of joint point ids (unique)
        """
        frame_info1 = self.get_frame_info(frame_id1)
        point_ids1 = set(frame_info1['point_ids'])
        list_point_ids = []

        for frame_id in frame_ids:
            frame_info = self.get_frame_info(frame_id)
            point_ids2 = frame_info['point_ids']
            list_point_ids.extend(point_ids2)

        list_point_ids = set(list_point_ids)  # remove duplicates

        # now find the points that are in frame_id1 and the list_point_ids
        joint_point_ids = list(point_ids1 & list_point_ids)
        return joint_point_ids

    def compute_scene_bounding_box(self):
        """
        Computes the axis-aligned bounding box of the scene points.
        Returns:
            dict: {
                "min": np.array([min_x, min_y, min_z]),
                "max": np.array([max_x, max_y, max_z]),
                "center": np.array([center_x, center_y, center_z]),
                "extent": float (norm of diagonal)
            }
        """

        if len(self.active_point_ids) == 0:
            return None

        xyzs = []

        for pid in self.active_point_ids:
            p3d = self.recon.points3D[pid]
            xyzs.append(p3d.xyz)

        xyz_arr = np.array(xyzs)
        min_bound = np.min(xyz_arr, axis=0)
        max_bound = np.max(xyz_arr, axis=0)
        center = (min_bound + max_bound) / 2.0
        extent = np.linalg.norm(max_bound - min_bound)

        return {
            "min": min_bound,
            "max": max_bound,
            "center": center,
            "extent": extent
        }

    def find_neighbor_frames(self, frame_id: int, min_points: int = 10) -> list[int]:
        """
        Finds neighbor frames for a given frame based on shared 3D points.
        Args:
            frame_id: ID of the frame to process
            min_points: Minimum number of shared points to consider an image similar
        Returns:
            list of neighbor frame IDs
        """
        
        frame_info = self.get_frame_info(frame_id)
        shared_point_counts = []

        for fid in self.frame_ids():
            if fid == frame_id:
                continue
            fid_info = self.get_frame_info(fid)
            n_shared_points = len(set(frame_info['point_ids']) & set(fid_info['point_ids']))
            if n_shared_points >= min_points:
                shared_point_counts.append((fid, n_shared_points))
        shared_point_counts.sort(key=lambda x: x[1], reverse=True)
        neighbor_frame_ids = [ fid for fid, _ in shared_point_counts ]
        return neighbor_frame_ids


    def render_points(self, image_id: int, target_size: int = 1024) -> np.ndarray | None:
        image_info = self.get_image_info(image_id)
        point_ids = image_info['point_ids']

        if len(point_ids) == 0:
            print(f"No points found for image {image_id}")
            return None

        img = cv2.imread(str(image_info['image_path']))
        if img is None:
            raise FileNotFoundError(f"Image {image_id} not found: {image_info['image_path']}")

        max_dim = max(img.shape[0], img.shape[1])
        target_width  = int(target_size * img.shape[1] / max_dim)
        target_height = int(target_size * img.shape[0] / max_dim)

        if target_width != img.shape[1] or target_height != img.shape[0]:
            img = cv2.resize(img, (target_width, target_height))

        K = image_info['K']
        # scale K for the target size
        scale_factor = target_size / max_dim
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_factor
        K_scaled[1, 1] *= scale_factor
        K_scaled[0, 2] *= scale_factor
        K_scaled[1, 2] *= scale_factor

        points_xyz = []
        for pid in point_ids:
            p3d = self.recon.points3D[pid]
            points_xyz.append(p3d.xyz)

        points_xyz = np.array(points_xyz)
        dist_params = image_info['distortion_params']

        # project points using K R t
        rvec = cv2.Rodrigues(image_info['R'])[0]
        tvec = image_info['t']
        points_uv = cv2.projectPoints(points_xyz, rvec, tvec, K_scaled, dist_params)[0]

        points_uv = points_uv.reshape(-1, 2)
        points_uv = points_uv.astype(np.int32)
        points_uv = points_uv[points_uv[:, 0] >= 0]
        points_uv = points_uv[points_uv[:, 0] < target_width]
        points_uv = points_uv[points_uv[:, 1] >= 0]
        points_uv = points_uv[points_uv[:, 1] < target_height]

        # draw points on the image
        for uv in points_uv:
            cv2.circle(img, (uv[0], uv[1]), 5, (0, 0, 255), -1)

        return img

    def save_point_cloud(self, output_path, axis_scale=None):
        # 1. Points 3D
        xyzs = []
        rgbs = []

        for pid in self.active_point_ids:
            p3d = self.recon.points3D[pid]
            xyzs.append(p3d.xyz)
            rgbs.append(p3d.color)

        if len(xyzs) == 0:
            print("No points found")
            return

        # Compute scene scale if axis_scale is not provided
        if axis_scale is None:
            axis_scale = np.clip(self.scene_bbox['extent'] * 0.01, 0.5, 10)

        self._write_ply_points(output_path, np.array(xyzs), np.array(rgbs))
        print(f"Saved point cloud to {output_path} [axis_scale={axis_scale}]")

        # 2. Cameras (Axes)
        vertices = []
        vertex_colors = []

        # Iterate over frames and draw only the first camera of each frame
        if self.recon.num_frames() > 0:
            for frame_id in self.recon.frames:
                try:
                    frame_calib = self.get_frame_info(frame_id)
                    image_ids = frame_calib['image_ids']

                    if not image_ids:
                        continue

                    for img_id in image_ids:
                        image_info = frame_calib['image_infos'][img_id]
                        v, c = self._draw_camera(image_info, axis_scale)
                        vertices.extend(v)
                        vertex_colors.extend(c)

                except Exception as e:
                     print(f"Error processing frame {frame_id}: {e}")

        else:
            # Fallback if no frames: draw all images
            for image_id, image in self.recon.images.items():
                image_info = self.get_image_info(image_id)
                v, c = self._draw_camera(image_info, axis_scale)
                vertices.extend(v)
                vertex_colors.extend(c)

        base, ext = os.path.splitext(output_path)
        cam_path = f"{base}_cameras{ext}"

        if vertices:
            # Reusing _write_ply_points since we are just saving points now
            self._write_ply_points(cam_path, np.array(vertices), np.array(vertex_colors))
            print(f"Saved camera axes points to {cam_path}")

    def _draw_camera(self, image_info: dict, axis_scale: float):
        vertices = []
        vertex_colors = []
        N = 10

        center = image_info['c']
        R = image_info['R']

        # Center point (White)
        vertices.append(center)
        vertex_colors.append([255, 255, 255])

        x_axis = R[0, :]
        y_axis = R[1, :]
        z_axis = R[2, :]

        for i in range(1, N + 1):
            scale = (i / N) * axis_scale

            # X axis (Red)
            vertices.append(center + x_axis * scale)
            vertex_colors.append([255, 0, 0])

            # Y axis (Green)
            vertices.append(center + y_axis * scale)
            vertex_colors.append([0, 255, 0])

            # Z axis (Blue)
            vertices.append(center + z_axis * scale)
            vertex_colors.append([0, 0, 255])

        return vertices, vertex_colors

    def _write_ply_points(self, path, points, colors):
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points)):
                p = points[i]
                c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    def transfer_to_face_mask(self, image_info: dict, face_mask: np.ndarray, frame_info: dict, eq_mask: np.ndarray, spot_width: int=3):
        """
        Transfer the face mask from the cubemap image to the equirectangular image. Operates over 0's of the face_mask and sets the corresponding 0's of the eq_mask.
        Args:
            image_info: dict of the image info
            face_mask: mask for the face of the cubemap image
            eq_mask: mask for the equirectangular image
            spot_width: Width of the spot to transfer the mask from the cubemap image to the equirectangular image
        """
        
        K = image_info['K']
        R = image_info['R']

        # uv coordinates for face_mask > 0:
        uv_coords = np.where(face_mask == 0)
        if len(uv_coords[0]) == 0:
            return
        # convert uv coordinates to rays in camera coordinates using K and R t
        K_inv = np.linalg.inv(K)
        rays_in_cam = K_inv @ np.stack([uv_coords[1], uv_coords[0], np.ones_like(uv_coords[0])], axis=0)
        rays_in_cam = rays_in_cam / np.linalg.norm(rays_in_cam, axis=0)
        rays_in_cam = rays_in_cam.T

        rays = rays_in_cam @ R @ frame_info['R'].T

        eq_height, eq_width = eq_mask.shape
        uv = spherical_img_from_cam((eq_width, eq_width/2), rays)
        uv = uv.astype(np.int32)
        mask = (uv[:, 0] >= spot_width) & (uv[:, 0] < eq_width - spot_width) & (uv[:, 1] >= spot_width) & (uv[:, 1] < eq_height - spot_width)
        uv = uv[mask]

        for u, v in uv:
            eq_mask[v-spot_width:v+spot_width, u-spot_width:u+spot_width] = 0

    def get_spherical_mask_from_cubemap(self, frame_id: int, eq_width: int, spot_width: int=3) -> np.ndarray:
        """
        Get the mask of the spherical image from the frame's cameras. The cubemap images are stored in the frame's image_ids.
        Args:
            frame_id: ID of the frame to process
            eq_width: Width of the equirectangular image (height = width / 2)
            spot_width: Width of the spot to transfer the mask from the cubemap image to the equirectangular image
        Returns:
            mask: (H, W) mask of the spherical image
        """

        if self.pano_face_mask_folder is None:
            raise ValueError("Pano face mask folder not found")
        
        frame_info = self.get_frame_info(frame_id)
        image_ids = frame_info['image_ids']
        if len(image_ids) != 6:
            raise ValueError(f"Frame {frame_id} has {len(image_ids)} images, expected 6")
        
        eq_mask = np.ones((eq_width//2, eq_width), dtype=np.uint8) * 255
        for img_id in image_ids:
            image_info = frame_info['image_infos'][img_id]
            face_mask_path = self.pano_face_mask_folder / f"{image_info['image_name']}.png"
            face_mask = cv2.imread(str(face_mask_path), cv2.IMREAD_GRAYSCALE)

            # face_mask is a mask for the face of the cubemap image
            # we need to convert it to a mask for the equirectangular image

            self.transfer_to_face_mask(image_info, face_mask, frame_info, eq_mask, spot_width)

        return eq_mask

    def spherical_frame_depth_samples(self, frame_id: int, eq_width: int) -> tuple[list[np.ndarray], list[int]]:
        """
        Generate depth samples for a spherical image from the frame's cameras.
        Args:
            frame_id: ID of the frame to process
            eq_width: Width of the equirectangular image (height = width / 2)
        Returns:
            depth_samples: Nx3 -> list[(u, v, depth)] samples in the equirectangular image
            sensor_ids: list of sensor IDs corresponding to each depth sample
        """

        frame_info = self.get_frame_info(frame_id)
        image_ids = frame_info['image_ids']
        if len(image_ids) != 6:
            raise ValueError(f"Frame {frame_id} has {len(image_ids)} images, expected 6")

        R = frame_info['R']
        t = frame_info['t']

        depth_samples = []
        sample_ids = []

        all_points_in_frame = []

        sensor_mapping = frame_info['sensor_mapping']
        for sensor_id, img_id in sensor_mapping.items():
            image_info = self.get_image_info(img_id)
            point_ids = image_info['point_ids']
            if len(point_ids) == 0:
                continue
            points_xyz = np.array([self.recon.points3D[pid].xyz for pid in point_ids]).reshape(-1, 3)
            points_in_frame = points_xyz @ R.T + t # [N, 3] world to camera coordinates
            all_points_in_frame.extend(points_in_frame)
            uv = spherical_img_from_cam((eq_width, eq_width/2), points_in_frame)
            depth = np.linalg.norm(points_in_frame, axis=1)

            for (u, v), d in zip(uv, depth):
                depth_samples.append(np.array([u, v, d]))
                sample_ids.append(sensor_id)

        all_points_in_frame = np.array(all_points_in_frame)

        return depth_samples, sample_ids, all_points_in_frame

    def render_spherical_frame(self, frame_id: int, eq_image: np.ndarray) -> np.ndarray | None:

        eq_width = eq_image.shape[1]
        depth_samples, sample_ids, _ = self.spherical_frame_depth_samples(frame_id, eq_width)

        keys = sorted(list(set(sample_ids)))

        colors = { sensor_id: color for sensor_id, color in zip(
            sorted(keys),
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255]
            ]
        )
        }

        for ds, sensor_id in zip(depth_samples, sample_ids):
            u = int(ds[0])
            v = int(ds[1])
            cv2.circle(eq_image, (u, v), 5, colors[sensor_id], -1)

        return eq_image


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--workfolder", "-w", type=Path, default="data")
    parser.add_argument("--model-folder", "-m", type=Path, default="sparse")
    parser.add_argument("--image-folder", "-i", type=Path, default="images_cubemap")
    parser.add_argument("--n-min-tracks", type=int, default=3)
    parser.add_argument("--output-folder", type=Path, default="output", help="Output folder relative to the model path")

    args = parser.parse_args()

    workfolder = Path(args.workfolder).resolve()
    output_folder = workfolder / args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    ci = ColmapInterface(workfolder, args.model_folder, args.image_folder, args.n_min_tracks)

    print(f"Info: {ci.get_info()}")

    image_ids = ci.image_ids()
    frame_ids = ci.frame_ids()

    if len(image_ids) > 0:
        first_img_id = image_ids[0]
        info = ci.get_image_info(first_img_id)
        print(f"Calibration for image {first_img_id}:")
        print(f"  Name: {info['image_name']}")
        print(f"  Path: {info['image_path']}")
        print(f"  K:\n{info['K']}")
        print(f"  Distortion: {info['distortion_params']}")
        print(f"  Point IDs: {len(info['point_ids'])}")

    if len(frame_ids) > 0:
        first_frame_id = frame_ids[0]
        frame_calib = ci.get_frame_info(first_frame_id)
        print(f"Calibration for frame {first_frame_id}:")
        print(f"  Keyframe path: {frame_calib['keyframe_path']}")
        print(f"  Rig ID: {frame_calib['rig_id']}")
        print(f"  Num images: {len(frame_calib['image_ids'])}")

    # Save point cloud
    ci.save_point_cloud(output_folder / "reconstruction.ply")

    count = 0
    for frame_id in frame_ids:
        count += 1
        if count > 20: # limit to first 10 frames for testing
            break
        frame_info = ci.get_frame_info(frame_id)
        keyframe_path = frame_info['keyframe_path']
        if not keyframe_path.exists():
            print(f"No keyframe path found for frame {frame_id}: {keyframe_path}")
            continue

        eq_image = cv2.imread(str(keyframe_path))
        if eq_image is None:
            print(f"Failed to read equirectangular image {keyframe_path}")
            continue
        eq_image = cv2.resize(eq_image, (2048, 1024))

        print(f"Projecting frame {frame_id} on equirectangular image")
        rimg = ci.render_spherical_frame(frame_id, eq_image)
        if rimg is not None:
            cv2.imwrite(output_folder / f"rendered-{frame_id:06d}.jpg", rimg)
        else:
            print(f"Failed to project points for frame {frame_id}")

def compute_processing_order(ci: ColmapInterface) -> list[int]:
    """
    Compute the processing order of the frames using the most connected frames first and then
    adding new frames by the number of shared points with the most connected frames. This is implemented 
    as a greedy algorithm.
    """

    # initialially generate a list of all frame ids with the number of visible points in each frame
    frame_ids = ci.frame_ids()

    frame_points = [ [fid, len(ci.get_frame_info(fid)['point_ids'])] for fid in frame_ids ]
    frame_points = sorted(frame_points, key=lambda x: x[1], reverse=True)

    frame_order = []
    point_th = frame_points[0][1] * 0.9

    # insert frames that have a similar number of points with the first frame into the frame_order array
    for frame_id, num_points in frame_points:
        if num_points > point_th:
            frame_order.append(frame_id)
        else:
            break

    remaining_frame_ids = [ fid for fid in frame_ids if fid not in frame_order ]

    # cache the joint point counts between frames to avoid recomputing them
    frame_to_frame_counts = {}
    pairs = []
    for fid1 in frame_ids:
        frame_to_frame_counts[fid1] = {}
        for fid2 in frame_ids:
            if fid2 <= fid1:
                continue
            pairs.append((fid1, fid2))

    def _get_joint_points(frame_pair: tuple[int, int]):
        fid1, fid2 = frame_pair
        frame_to_frame_counts[fid1][fid2] = len(ci.get_joint_points(fid1, fid2))
        frame_to_frame_counts[fid2][fid1] = frame_to_frame_counts[fid1][fid2]

    parallel_executor = ParallelExecutor()
    parallel_executor.run_in_parallel_no_return(_get_joint_points, item_list=pairs, progress_desc="Computing joint point counts")

    # now iterate over the frame points and add new frames by the number of shared points with the most connected frames
    while len(remaining_frame_ids) > 0:

        if len(remaining_frame_ids) < 5:
            frame_order.extend(remaining_frame_ids)
            break

        # compute the joint point counts with the frames already added to the frame_order array
        joint_point_counts = [ [fid, sum(frame_to_frame_counts[fid][other_fid] for other_fid in frame_order)] for fid in remaining_frame_ids ]
        joint_point_counts = sorted(joint_point_counts, key=lambda x: x[1], reverse=True)

        # add the frame with the most shared points to the frame_order array
        point_count_th = joint_point_counts[0][1] * 0.9
        for fid, point_count in joint_point_counts:
            if point_count > point_count_th:
                frame_order.append(fid)
                remaining_frame_ids.remove(fid)
            else:
                break

    return frame_order

if __name__ == "__main__":
    main()

