# [Temporal-Spatial Object Relations Modeling for Vision-and-Language Navigation](https://ieeexplore.ieee.org/document/11141526)

## Prerequisites
1. Our model was trained and evaluated using the following package dependencies:
* Pytorch 2.1.1
* Python 3.9.18

2. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).

3. Download data from [here](https://pan.baidu.com/s/1fvxyd9o2rKM1TNIyNm7Lcg?pwd=mgdj), Put the data in `datasets` directory.

## Pre-training
```
cd pretrain_src
bash pretrain.sh
```
## Fine-tuning
```
cd main_src
bash scripts/train.sh 8001
```
## Citation
If you find this work useful in your research, please cite the following paper:
```
# BibTeX
@ARTICLE{11141526,
  author={Huang, Bowen and Zheng, Yanwei and Sui, Dongchen and Lan, Chuanlin and Zhao, Xinpeng and Zhang, Xiao and Meng, Jingke and Xiao, Mengbai and Zou, Yifei and Yu, Dongxiao},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Temporal-Spatial Object Relations Modeling for Vision-and-Language Navigation}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Navigation;Training;Feature extraction;Visualization;Transformers;Trajectory;Turning;Planning;Reinforcement learning;Large language models;Vision-and-language navigation;temporal object relations;spatial object relations;turning back penalty},
  doi={10.1109/TITS.2025.3599649}}

# GB/T 7714
[1] Huang B , Zheng Y , Lan C ,et al.Temporal-Spatial Object Relations Modeling for Vision-and-Language Navigation[J].  2024.DOI:10.1109/TITS.2025.3599649.

# MLA
[1] Huang, Bowen , et al. "Temporal-Spatial Object Relations Modeling for Vision-and-Language Navigation." (2024).

# APA
[1] Huang, B. ,  Zheng, Y. ,  Lan, C. ,  Zhao, X. ,  Zou, Y. , &  Yu, D. . (2024). Temporal-spatial object relations modeling for vision-and-language navigation.

## Acknowledgement
Codebase from [DUET](https://github.com/cshizhe/VLN-DUET).
