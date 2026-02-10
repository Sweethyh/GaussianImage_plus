# GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![arXiv](https://img.shields.io/badge/GaussianImage_plus-2512.19108-b31b1b)](https://arxiv.org/abs/2512.19108)
[![GitHub Repo stars](https://img.shields.io/github/stars/Sweethyh/GaussianImage_plus.svg?style=social&label=Star&maxAge=60)](https://github.com/Sweethyh/GaussianImage_plus)

[[paper](https://arxiv.org/abs/2512.19108)][[code](https://github.com/Xinjie-Q/GaussianImage)]

[Tiantian Li](https://sweethyh.github.io/), [Xinjie Zhang](https://xinjie-q.github.io/), [Xingtong Ge](https://xingtongge.github.io/), [Tongda Xu](https://tongdaxu.github.io/), [Dailan He](https://scholar.google.com/citations?user=f5MTTy4AAAAJ&hl=en), [Jun Zhang](https://eejzhang.people.ust.hk/), [Yan Wang📧](https://yanwang202199.github.io/)

(📧 denotes corresponding author.)

This is the official implementation of our paper [GaussianImage++](https://arxiv.org/abs/2512.19108), accepted by AAAI 2026.



## News

* **2025/12/23**: 🔥 We release our Python and CUDA code for GaussianImage++ presented in our paper. Have a try! 
<!-- Compared to the first version, we further improved the decoding speed of the GaussianImage codec to around 2000 FPS by removing the entropy coding operation, while only increasing the bpp overhead very slightly. -->

* **2025/11/8**: 🌟 Our paper has been accepted by AAAI 2026! 🎉 Cheers!

## Quick Started

### Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:Sweethyh/GaussianImage_plus.git 
```
or
```shell
# HTTPS
git clone https://github.com/Sweethyh/GaussianImage_plus.git 
```
After cloning the repository, you can follow these steps to train GaussianImage++ models under different tasks. 

### Requirements

```bash
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

If you encounter errors while installing the packages listed in requirements.txt, you can try installing each Python package individually using the pip command.

Before training, you need to download the [kodak](https://r0k.us/graphics/kodak/) and [DIV2K-validation](https://data.vision.ee.ethz.ch/cvl/DIV2K/) datasets. The dataset folder is organized as follows.

```bash
├── dataset
│   | kodak 
│     ├── kodim01.png
│     ├── kodim02.png 
│     ├── ...
│   | DIV2K_valid_HR
│     ├── 0801.png
│     ├── 0802.png
│     ├── 0803.png
│     ├── ...
```

#### Representation

```bash
python train.py --num_points 2500 --max_num_points 5000 --data_name kodak -d ./dataset/kodak/
```

#### Compression

```bash
python train.py --num_points 2500 --max_num_points 5000 --data_name kodak -d ./dataset/kodak/ --color_norm 

python train_quantize.py --num_points 2500 --max_num_points 5000 --data_name kodak -d ./dataset/kodak/ --color_norm 
```

## Acknowledgments

Our code was developed based on [GaussianImage](https://github.com/Xinjie-Q/GaussianImage). We thank them for providing the novel framework to implement image representation and compression.

```
