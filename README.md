# GPDRNet

Code for **Robust Pixel-by-Pixel Multimodal Remote Sensing Image Registration Using Geometry Preserving Dense Registration Network**. (TGRS 2026)

[Paper](https://doi.org/10.1109/TGRS.2026.3664764)

## Citation

If you find this work useful, please cite:

@ARTICLE{11396038,
  author={Zhou, Liang and Peng, Tao and Han, Zhiqiang and Li, Liangzhi and Zhu, Qing and Ye, Yuanxin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Robust Pixel-by-Pixel Multimodal Remote Sensing Image Registration Using Geometry Preserving Dense Registration Network}, 
  year={2026},
  volume={64},
  number={},
  pages={1-14},
  keywords={Deformation;Feature extraction;Image registration;Geometry;Synthetic aperture radar;Optical sensors;Optical flow;Estimation;Accuracy;Remote sensing;Dense registration;geometry preservation;global deformation;local nonrigid deformation;multimodal remote sensing (RS) image registration},
  doi={10.1109/TGRS.2026.3664764}}

## Abstract

The integration of multimodal remote sensing (RS) data is pivotal for advancing Earth observation capabilities, where high-precision cross-modal image registration serves as the cornerstone for realizing its full potential. While sparse feature matching methods have achieved high registration accuracy by extracting corresponding points and solving the parameters of the assumed geometric transformation model, they still struggle to effectively address local nonrigid deformations between multimodal images. Additionally, image dense registration methods are prone to introducing significant mismatching noise, which frequently results in structural distortions of typical ground features (e.g., roads, buildings, etc.) within global deformation regions. To address that, we propose a novel geometry-preserving dense registration network (GPDRNet) for multimodal RS image registration tasks. Our approach incorporates three key innovations: 1) a feature enhancement module that generates more discriminative cross-modal feature representations; 2) a geometry consistency loss that preserves critical structural and morphological characteristics of typical ground features within global deformation regions; and 3) a local deformation-aware postprocessing module that first estimates a global affine field, and then identifies local deformation regions through residual flow analysis. Experimental results validate the superior effectiveness and robustness of the proposed GPDRNet over existing state-of-the-art multimodal RS image registration methods.

## File Preparation

Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1Uf7cGK8wSI4VEuQvOqQjBi3G6CVKB27E?usp=drive_link). Place the checkpoint files in `weights/`, keeping the filenames shown below unchanged.

Place paired optical and SAR images directly in `dataset/`, using the same index for each pair:

```text
GPDRNet-release/
├── register.py
├── requirements.txt
├── gpdrnet/
├── weights/
│   ├── registration.pth
│   ├── dinov2_vitl14_pretrain.pth
│   └── ldapp.pth
├── dataset/
│   ├── optical_01.tif
│   ├── sar_01.tif
│   ├── optical_02.tif
│   ├── sar_02.tif
│   └── ...
└── results/
```

## Run Registration

Run the following command from the release directory to process the first pair on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python register.py \
  --sar dataset/sar_01.tif \
  --optical dataset/optical_01.tif \
  --output results/run_01 \
  --device cuda:0
```
