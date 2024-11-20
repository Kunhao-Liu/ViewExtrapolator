# [*arXiv 2024*] ViewExtrapolator: Novel View Extrapolation with Video Diffusion Priors

## [Project page](https://kunhao-liu.github.io/ViewExtrapolator/) |  [Paper]()

<p float="left" align="center">
  <img src="https://kunhao-liu.github.io/ViewExtrapolator/assets/images/ninjabike.gif" width="23%" />
  <img src="https://kunhao-liu.github.io/ViewExtrapolator/assets/images/orchid.gif" width="23%" /> 
  <img src="https://kunhao-liu.github.io/ViewExtrapolator/assets/images/caterpillar.gif" width="23%" />
  <img src="https://kunhao-liu.github.io/ViewExtrapolator/assets/images/hike.gif" width="23%">
</p>


This repository contains the official implementation of the paper: [Novel View Extrapolation with Video Diffusion Priors](). We introduce ViewExtrapolator, a novel approach that leverages the generative priors of Stable Video Diffusion for novel view extrapolation, where the novel views lie far beyond the range of the training views.



## To Begin
Our codes are tested on `python=3.11, pytorch=2.2.0, CUDA=12.1`.

1. Clone ViewExtrapolator.
```bash
git clone https://github.com/Kunhao-Liu/ViewExtrapolator.git
cd ViewExtrapolator
```

2. Please refer to the [multiview folder](https://github.com/Kunhao-Liu/ViewExtrapolator/tree/main/multiview) for novel view extrapolation with 3D Gaussian Splatting when multiview images are available.

3. Please refer to the [monocular folder](https://github.com/Kunhao-Liu/ViewExtrapolator/tree/main/monocular) for novel view extrapolation with point clouds when only a single view or monocular video is available.

## Acknowledgements

Our work is based on [Stable Video Diffusion](https://stability.ai/stable-video) and [gsplat](https://github.com/nerfstudio-project/gsplat) implementation of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) . We thank the authors for their great work and open-sourcing the code.

## Citation
Consider citing us if you find this project helpful.
```
@article{liu2024novel,
  title   = {Novel View Extrapolation with Video Diffusion Priors},
  author  = {Liu, Kunhao and Shao, Ling and Lu, Shijian},
  journal = {arXiv preprint arXiv:???},
  year    = {2024}
}
```
