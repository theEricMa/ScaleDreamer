# ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation

### 📢 News

- **2024.06.23** Create this repo.

## ⚙️ Dependencies and Installation
You must have an NVIDIA graphics card with at least [?]GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed. Follow the [threestudio](https://github.com/threestudio-project/threestudio) instructions to set up your environment, or use the instructions below.
- Create a virtual environment:

```sh
conda create -n scaledreamer python=3.11
conda activate scaledreamer
```
- Install PyTorch
```sh
# Prefer using the latest version of CUDA and PyTorch for compatibility with xformers
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
- (Optional, Recommended) Install [xFormers](https://github.com/facebookresearch/xformers) for Attention acceleration.
```sh
conda install xformers -c xformers
```
- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install major dependencies:

```sh
pip install -r requirements.txt
```

- Install [xformer](https://github.com/facebookresearch/xformers#installing-xformers), assume the CUDA version is cu118.
```sh
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```
- Install iNGP dependencies (according to your default CUDA):

```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
