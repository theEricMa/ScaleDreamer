# ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation
## Paper

## Installation

See [installation.md](docs/installation.md) for additional information, including installation via Docker.


- You must have an NVIDIA graphics card with at least 6GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.

```sh
# mostly prefer torch 2.2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
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
