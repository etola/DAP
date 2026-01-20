import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2
import os
import numpy as np

from ddpy_dap import infer as infer_module

load_model = infer_module.load_model
infer_raw = infer_module.infer_raw
pred_to_vis = infer_module.pred_to_vis

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

def process_dataset(conf: dict):

    ml_dmap_folder = conf["ml_dmap_folder"]
    ml_dmap_folder.mkdir(parents=True, exist_ok=True)

    # expecting image folder for equirectangular images
    image_folder = conf["image_folder"]
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    depth_scale = 10 if conf["vis"] == "10m" else 100

    # Load Config and Model
    with open(conf["config"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model, device = load_model(config)

    infer_width = conf["infer_width"]

    # get all files under images directory
    img_files = sorted(list(conf["image_folder"].glob("*.jpg")))

    # split img_files into batches
    img_batches = [img_files[i:i+conf["batch_size"]] for i in range(0, len(img_files), conf["batch_size"])]

    for img_batch in tqdm(img_batches, desc="Processing"):

        img_inputs = []
        for img_path in img_batch:
            img_input = load_image(img_path, infer_width, infer_width//2)
            img_inputs.append(img_input)
            img_name = img_path.name.split('.')[0]

        img_inputs = np.stack(img_inputs, axis=0)
        pred_depths = infer_raw(model, device, img_inputs)
        for i in range(len(img_batch)):
            pred_depth = pred_depths[i] * depth_scale
            img_name = img_batch[i].name.split('.')[0]
            np.save(ml_dmap_folder / f"{img_name}.npy", pred_depth)

            if conf["save_visualization"]:
                _, depth_color_rgb = pred_to_vis(pred_depth, vis_range=conf["vis"], cmap=conf["cmap"])
                vis_path = os.path.join(ml_dmap_folder, f"{img_name}.png")
                cv2.imwrite(vis_path, cv2.cvtColor(depth_color_rgb, cv2.COLOR_RGB2BGR))

                # cam_points = depth_map_to_cam_points(pred_depth)
                # mask = pred_depth > 0
                # cam_points = cam_points[mask].reshape(-1,3)
                # colors  = img_inputs[i][mask].reshape(-1,3)
                # save_point_cloud(cam_points, colors, out_pts / f"{img_name}-raw.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=Path, help="Path to the dataset folder")
    parser.add_argument("--image_folder", "-i", type=Path, default="keyframes", help="Path to the image folder relative to dataset. default [keyframes]")
    parser.add_argument("--output", "-o", type=Path, default="./", help="Path to the output folder relative to dataset. default [./]")
    parser.add_argument("--config", '-c', default="config/infer.yaml", help="Path to inference config, default [config/infer.yaml]")
    parser.add_argument("--vis", default="10m", choices=["100m", "10m"], help="Visualization range, default [10m]")
    parser.add_argument("--cmap", default="Spectral", help="Colormap name, e.g. default [Spectral], Turbo, Viridis")
    parser.add_argument("--save_visualization", "-v", action="store_true", help="Save visualizations (default: False) saves dmap and pointcloud")
    parser.add_argument("--infer_width", type=int, default=1024, help="Inference width, default [1024]")
    parser.add_argument("--batch_size", "-b", type=int, default=12, help="Batch size, default [12]")

    args = parser.parse_args()

    # if image_folder is an absolute path, use it as is
    if args.image_folder.is_absolute():
        image_folder = args.image_folder
    else:
        image_folder = (args.dataset / args.image_folder).resolve()

    if args.output.is_absolute():
        output_folder = args.output
    else:
        output_folder = (args.dataset / args.output).resolve()

    conf = {
        "dataset": args.dataset,
        "out_folder": output_folder,
        "ml_dmap_folder": output_folder / "ml_dmaps",
        "image_folder": image_folder,
        "infer_width": args.infer_width,
        "config": args.config,
        "vis": args.vis,
        "cmap": args.cmap,
        "save_visualization": args.save_visualization,
        "batch_size": args.batch_size,
    }

    process_dataset(conf)

