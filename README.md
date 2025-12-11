# Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation

<p align="center">
  <a href='https://arxiv.org/abs/2510.11712'><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://fenghora.github.io/DiT360-Page/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=insta360&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/spaces/Insta360-Research/DiT360'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
  <a href='https://huggingface.co/datasets/Insta360-Research/Matterport3D_polished'><img src='https://img.shields.io/badge/%F0%9F%93%88%20Hugging%20Face-Dataset-yellow'></a>
</p>

![teaser](assets/depth_teaser2_00.png)



## 🔨 Installation

Clone the repo first:

```Bash
git clone https://github.com/Insta360-Research-Team/DAP
cd DAP
```

(Optional) Create a fresh conda env:

```Bash
conda create -n dap python=3.12
conda activate dap
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version, we test our code on torch==2.6.0 and torchvision==0.21.0)
pip install torch==2.6.0 torchvision==0.21.0

# other dependencies
pip install -r requirements.txt
```

## 🖼️ Dataset


## 📒 Inference

For a quick use, you can just try:

```Bash
python inference.py
```


## 🚀 Test

### Panorama Training

We provide a training pipeline based on **Insta360-Research/Matterport3D_polished**, along with corresponding launch scripts.
You can start training with a single command:


```bash
bash train.sh
```


After training is completed, you will find a checkpoint file saved under the output directory, typically like:


```bash
model_saved/lightning_logs/version_x/checkpoints/vsclip_epoch=xxx.ckpt/checkpoint/mp_rank_00_model_states.pt
```


You can extract the LoRA weights from the full `.pt` checkpoint by running:


```bash
python get_lora_weights.py <path_to_your_pt_file> <output_dir>
```


If you don’t specify `output_dir`, the extracted weights will be saved by default to:


```bash
lora_output/
```


After that, you can directly use your trained LoRA in the inference script.
Simply replace the default model path `"fenghora/DiT360-Panorama-Image-Generation"` in `inference.py` with your output directory (e.g., `"lora_output"`), and then run:


```bash
python inference.py
```


## 🤝 Acknowledgement

We appreciate the open source of the following projects:

* [diffusers](https://github.com/huggingface/diffusers)
* [Personalize Anything](https://github.com/fenghora/personalize-anything)
* [RF-Inversion](https://github.com/LituRout/RF-Inversion)
* [RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit)

## Citation
```
@misc{dit360,
  title={DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training}, 
  author={Haoran Feng and Dizhe Zhang and Xiangtai Li and Bo Du and Lu Qi},
  year={2025},
  eprint={2510.11712},
  archivePrefix={arXiv},
}
```
If you find our **dataset** useful, please include a citation for **Matterport3D**:
```
@article{Matterport3D,
  title={Matterport3D: Learning from RGB-D Data in Indoor Environments},
  author={Chang, Angel and Dai, Angela and Funkhouser, Thomas and Halber, Maciej and Niessner, Matthias and Savva, Manolis and Song, Shuran and Zeng, Andy and Zhang, Yinda},
  journal={International Conference on 3D Vision (3DV)},
  year={2017}
}
```
If you find our **inpainting & outpainting** useful, please include a citation for **Personalize Anything**:
```
@article{feng2025personalize,
  title={Personalize Anything for Free with Diffusion Transformer},
  author={Feng, Haoran and Huang, Zehuan and Li, Lin and Lv, Hairong and Sheng, Lu},
  journal={arXiv preprint arXiv:2503.12590},
  year={2025}
}
```
