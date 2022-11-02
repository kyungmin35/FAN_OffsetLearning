# Heatmap Correction with Offset Learning via Graph Neural Network for Face Alignment
## Introduction
We propose a hybrid form regression network that leverage both advantages of heatmap and coordinate regression methods in the way of coarse-to-fine strategy. Proposed method adopts graph neural network as a refinement module to guide each point with most correlated neighbor nodes that we define by analyzing correlation using canonical correlation analysis (CCA). Refinement network learns to predict the offset of each landmark, used to compensate poorly detected coordinates roughly induced from heatmap regression. Our method showed significant refinement ability in challenging cases in WFLW, 300W benchmarks, which verifies the robustness and effectiveness.

<img width="397" alt="image" src="https://user-images.githubusercontent.com/83903071/199401608-f70cfbca-4520-40f2-9d13-5b061356ff84.png">

## Results 
### WFLW

| NME |  *test* | *pose* | *illumination* | *occlution* | *blur* | *makeup* | *expression* |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|HRNetV2-W18 | 4.60 | 7.86 | 4.57 | 5.42 | 5.36 | 4.26 | 4.78 |
|Ours | 4.41 | 7.55 | 4.33 | 5.21 | 5.04 | 4.29 | 4.65 |

### 300W

| NME | *common*| *challenge* | *full* |
|:--:|:--:|:--:|:--:|:--:|
|HRNetV2-W18 | 2.91 | 5.11 | 3.34 |
| Ours | 2.92 | 5.04 | 3.33 |

## Environment
1. Python 3.7
2. CUDA 10.2, Cudnn 7.6.5 Pytorch 1.4, Torchvision 0.5.0


## Datasets
- 300W
- WFLW
- COFW

