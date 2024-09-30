<h1>MHCDIFF</h1>

## Multi-hypotheses Conditioned Point Cloud Diffusion for 3D Human Reconstruction from Occluded Image (NeurIPS 2024)

[Donghwan Kim](https://donghwankim0101.github.io/)<sup>1</sup>, [Tae-Kyun (T-K) Kim](https://sites.google.com/view/tkkim/home)<sup>1,2</sup>

<sup>1 </sup>KAIST, <sup>2</sup> Imperial College London

**[\[Project Page\]](https://donghwankim0101.github.io/projects/mhcdiff/) [\[Paper\]](https://arxiv.org/abs/2409.18364)**

> 3D human shape reconstruction under severe occlusion due to human-object or human-human interaction is a challenging problem. Parametric models i.e. SMPL(- X), which are based on the statistics across human shapes, can represent whole human body shapes but are limited to minimally-clothed human shapes. Implicit- function-based methods extract features from the parametric models to employ prior knowledge of human bodies and can capture geometric details such as clothing and hair. However, they often struggle to handle misaligned parametric models and inpaint occluded regions given a single RGB image. In this work, we propose a novel pipeline, MHCDIFF, Multi-hypotheses Conditioned Point Cloud Diffusion, composed of point cloud diffusion conditioned on probabilistic distributions for pixel-aligned detailed 3D human reconstruction under occlusion. Compared to previous implicit-function-based methods, the point cloud diffusion model can capture the global consistent features to generate the occluded regions, and the denoising process corrects the misaligned SMPL meshes. The core of MHCDIFF is extracting local features from multiple hypothesized SMPL(-X) meshes and aggregating the set of features to condition the diffusion model. In the experiments on CAPE and MultiHuman datasets, the proposed method outperforms various SOTA methods based on SMPL, implicit functions, point cloud diffusion, and their combined, under synthetic and real occlusions.

# Updates

-   2024/09/30: Code released.

# Installation

Please follow [ICON](https://github.com/YuliangXiu/ICON/blob/master/docs/installation.md) to initialize the environment, download the extra data, such as HPS and SMPL (using `fetch_hps.sh` and `fetch_data.sh`).

```bash
pip install -r requirements.txt
```

The 130 line of `config/structured.py` need to be the global path of the directory, for example `/home/{user}/MHCDiff`.

Download smpl files and pretrained model for [ProPose](https://github.com/NetEase-GameAI/ProPose) and place them at `hps/ProPose/model_files`

# Train

Sampling points from the raw scan.

```bash
python fps_sampling_thuman.py
```

Train the model

```bash
python main.py run.name={name} dataset.hps=gt
```

# Demo

Prepare demo images in `data/demo/images/*.png`

```bash
python main.py run.name={name} run.job=sample chekpoint.resume={global path for pretrained checkpoint} dataset.type=demo
```

# Inference

## CAPE dataset

Random mask the CAPE dataset

```
python random_masking.py
```

```bash
python main.py run.name={name} run.job=sample checkpoint.resume={global path for pretrained checkpoint}
```

## MultiHuman dataset

Download [MultiHuman dataset](https://github.com/y-zheng18/MultiHuman-Dataset)

The Original Structure (MultiHuman.zip)

```
MultiHuman
├──single
├──single_occluded
├──three
├──two_closely_inter
├──two_naturally_interactive
```

Place the folder on `data/MultiHuman`

Render the scan data to image

```bash
bash render_multi.sh
```

Inference

```bash
python main.py run.name={name} run.job=sample checkpoint.resume={global path for pretrained checkpoint} dataset.type=multihuman dataset.category=single

python main.py run.name={name} run.job=sample checkpoint.resume={global path for pretrained checkpoint} dataset.type=multihuman dataset.category=single_occluded

python main.py run.name={name} run.job=sample checkpoint.resume={global path for pretrained checkpoint} dataset.type=multihuman dataset.category=three

python main.py run.name={name} run.job=sample checkpoint.resume={global path for pretrained checkpoint} dataset.type=multihuman dataset.category=two_closely_inter

python main.py run.name={name} run.job=sample checkpoint.resume={global path for pretrained checkpoint} dataset.type=multihuman dataset.category=two_naturally_interactive
```

# Citation

If you find this work useful, please consider citing our paper.

```
@article{mhcdiff2024,
    author = {Kim, Donghwan and Kim, Tae-Kyun},
    title = {Multi-hypotheses Conditioned Point Cloud Diffusion for 3D Human Reconstruction from Occluded Image},
    journal = {arXiv preprint arXiv:2409.18364},
    year = {2024}
}
```

# Acknowledgements

-   Parts of our code are based on [PC^2](https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion), [ICON](https://github.com/YuliangXiu/ICON).

-   [smplx](https://github.com/vchoutas/smplx), [ProPose](https://github.com/NetEase-GameAI/ProPose), [PIXIE](https://github.com/YadiraF/PIXIE), for Human Pose & Shape Estimation

-   [CAPE](https://github.com/qianlim/CAPE) and [THuman](https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset) for Dataset

-   The [PyTorch3D](https://github.com/facebookresearch/pytorch3d) library
