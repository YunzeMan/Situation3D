# Situational Awareness Matters in 3D Vision Language Reasoning

<a href="https://yunzeman.github.io/" style="color:blue;">Yunze Man</a> ·
<a href="https://cs.illinois.edu/about/people/department-faculty/lgui" style="color:blue;">Liang-Yan Gui</a> ·
<a href="https://yxw.web.illinois.edu/" style="color:blue;">Yu-Xiong Wang</a>

[CVPR 2024] [[`Project Page`](https://yunzeman.github.io/situation3d/)] [[`arXiv`](https://arxiv.org/abs/2406.07544)] [[`pdf`](https://yunzeman.github.io/situation3d/static/Situation3D_CVPR2024.pdf)] [[`BibTeX`](#BibTex)]

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) [![arXiv](https://img.shields.io/badge/arXiv-2406.07544-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2406.07544) [![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://yunzeman.github.io/situation3d/) [![YouTube](https://img.shields.io/badge/YouTube-Video-F83323?style=flat&logo=youtube&logoColor=F83323)](https://www.youtube.com/watch?v=IvjZXOs0Ozo) [![GitHub](https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white)](https://github.com/YunzeMan/Situation3D) [![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper "Situational Awareness Matters in 3D Vision Language Reasoning" (CVPR 2024). The paper is available on [arXiv](https://arxiv.org/abs/2406.07544). The project page is online at [here](https://yunzeman.github.io/situation3d/).


## About
<img src="assets/SIG3D.png" width="100%"/>
Previous methods perform direct 3D vision language reasoning without modeling the situation of an embodied agent in the 3D environment. Our method, SIG3D, grounds the situational description in the 3D space, and then re-encodes the visual tokens from the agent's intended perspective before vision-language fusion, resulting in a more comprehensive and generalized 3D vision language (3DVL) representation and reasoning pipeline.


## Environmet Setup and Dataset Preparation
Please install the required packages and dependencies according to the environment.yml.

In addition,
- in order to use the SQA3D dataset, please follow [this repo](https://github.com/SilongYong/SQA3D) to download the dataset and necessary toolsets.
- in order to use the ScanQA dataset, please follow [this repo](https://github.com/ATR-DBI/ScanQA) to download the dataset and necessary toolsets.
- in order to use the 3D-LLM backbone model, please follow [this repo](https://github.com/UMass-Foundation-Model/3D-LLM) to install necessary dependencies and download the pre-trained model.

Finally, please download the ScanNet dataset from the official website and follow the instructions [here](https://github.com/pengsongyou/openscene/blob/main/scripts/preprocess) to preprocess the ScanNet dataset and get RGB video frames and point clouds for each scannet scene.

#### The latest code implementation and models will be out recently. Please stay tuned.




## BibTeX
If you use our work in your research, please cite our publication:
```bibtex
@inproceedings{man2024situation3d,
      title={Situational Awareness Matters in 3D Vision Language Reasoning},
      author={Man, Yunze and Gui, Liang-Yan and Wang, Yu-Xiong},
      booktitle={CVPR},
      year={2024} 
      }
```

## Acknowledgements
This repo is built based on the fantastic work [SQA3D](https://github.com/SilongYong/SQA3D), [ScanQA](https://github.com/ATR-DBI/ScanQA), and [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM/tree/main). We thank the authors for their great work and open-sourcing their codebase. 
