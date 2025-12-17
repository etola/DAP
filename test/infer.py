from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import yaml
import numpy as np
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import matplotlib

from networks.models import *


def colorize_depth(depth: np.ndarray, cmap: str = 'Spectral', depth_truncation: float = 1.0) -> np.ndarray:

    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    depth_normalized = np.clip(depth_normalized, 0, depth_truncation)
    
    depth_normalized = depth_normalized / depth_truncation
    
    colored = matplotlib.colormaps[cmap](depth_normalized)[..., :3]
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def load_model(config):
    model_path = os.path.join(config["load_weights_dir"], 'model.pth')
    print(f"Loading model from: {model_path}")
    model_dict = torch.load(model_path)

    model = make(config['model'])
    if any(key.startswith('module') for key in model_dict.keys()):
        model = nn.DataParallel(model)
        
    model.cuda()
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
    model.eval()
    print("Model loaded successfully\n")
    return model


def load_image(img_path, height=512, width=1024):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    
    return tensor, img.shape[:2]  


def infer_single_image(model, img_path, save_dir, height=512, width=1024, resize_back=False):
    img_tensor, original_size = load_image(img_path, height, width)
    img_tensor = img_tensor.cuda()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
        outputs['pred_mask'] = 1 - outputs['pred_mask']
        outputs['pred_mask'] = (outputs['pred_mask'] > 0.5)
        outputs['pred_depth'][~outputs['pred_mask']] = 1
        
        pred_depth = outputs['pred_depth'][0].detach().cpu().squeeze().numpy()
    
    if resize_back:
        pred_depth = cv2.resize(pred_depth, (original_size[1], original_size[0]), 
                               interpolation=cv2.INTER_LINEAR)
    
    depth_colored = colorize_depth(pred_depth)
    
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    vis_path = os.path.join(save_dir, f'{img_name}.png')
    
    cv2.imwrite(vis_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
    
    return vis_path


def infer_from_list(model, txt_path, save_dir, height=512, width=1024, resize_back=False):
    with open(txt_path, 'r') as f:
        img_list = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Total images to process: {len(img_list)}\n")
    
    pbar = tqdm.tqdm(img_list)
    pbar.set_description("Inferencing")
    
    success_count = 0
    for img_path in pbar:
        try:
            vis_path = infer_single_image(model, img_path, save_dir, height, width, resize_back)
            success_count += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
    
    print(f"\n✓ Successfully processed {success_count}/{len(img_list)} images")
    print(f"✓ Results saved to: {save_dir}")


def infer_from_folder(model, input_dir, save_dir, height=512, width=1024, resize_back=False):
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    img_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file.lower())[1] in img_extensions:
                img_list.append(os.path.join(root, file))
    
    print(f"Found {len(img_list)} images in {input_dir}\n")
    
    pbar = tqdm.tqdm(img_list)
    pbar.set_description("Inferencing")
    
    success_count = 0
    for img_path in pbar:
        try:
            vis_path = infer_single_image(model, img_path, save_dir, height, width, resize_back)
            success_count += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
    
    print(f"\n✓ Successfully processed {success_count}/{len(img_list)} images")
    print(f"✓ Results saved to: {save_dir}")


def main(config):

    model = load_model(config)
    
    save_dir = config.get('save_depth_dir', 'depth_results')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving depth visualizations to: {save_dir}\n")
    
    infer_config = config.get('inference', {})
    height = infer_config.get('height', 512)
    width = infer_config.get('width', 1024)
    resize_back = infer_config.get('resize_back', False)
    
    if 'image_path' in infer_config:
        img_path = infer_config['image_path']
        print(f"Processing single image: {img_path}")
        vis_path = infer_single_image(model, img_path, save_dir, height, width, resize_back)
        print(f"\n✓ Result saved to: {vis_path}")
        
    elif 'image_list' in infer_config:
        txt_path = infer_config['image_list']
        print(f"Processing images from list: {txt_path}")
        infer_from_list(model, txt_path, save_dir, height, width, resize_back)
        
    elif 'input_dir' in infer_config:
        input_dir = infer_config['input_dir']
        print(f"Processing images from folder: {input_dir}")
        infer_from_folder(model, input_dir, save_dir, height, width, resize_back)
        
    else:
        print("Error: No input specified in config!")
        print("Please specify one of: 'image_path', 'image_list', or 'input_dir'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Depth Estimation Inference')
    parser.add_argument('--config', default='config/infer.yaml', help='Path to inference config file')
    parser.add_argument('--gpu', default='0', help='GPU device ID')
    parser.add_argument('--image', default=None, help='Single image path (overrides config)')
    parser.add_argument('--image_list', default=None, help='Text file with image paths (overrides config)')
    parser.add_argument('--input_dir', default=None, help='Input directory (overrides config)')
    parser.add_argument('--output', default=None, help='Output directory (overrides config)')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('Config loaded.\n')
    
    if 'inference' not in config:
        config['inference'] = {}
    
    if args.image is not None:
        config['inference']['image_path'] = args.image
    if args.image_list is not None:
        config['inference']['image_list'] = args.image_list
    if args.input_dir is not None:
        config['inference']['input_dir'] = args.input_dir
    if args.output is not None:
        config['save_depth_dir'] = args.output
    
    main(config)
