# TSOR

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

## Acknowledgement
Codebase from [DUET](https://github.com/cshizhe/VLN-DUET).
