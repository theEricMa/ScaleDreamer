# ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation

### 📢 News

- **2024.06.23** Create this repo.

## ⚙️ Dependencies and Installation
You must have an NVIDIA graphics card with at least [?]GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed. Follow the [threestudio](https://github.com/threestudio-project/threestudio) instructions to set up your environment, or use the instructions below.
- Create a virtual environment:

```sh
conda create -n scaledreamer python=3.10
conda activate scaledreamer
```
- Install PyTorch
```sh
# Prefer using the latest version of CUDA and PyTorch 
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
- (Optional, Recommended) Install [xFormers](https://github.com/facebookresearch/xformers) for attention acceleration.
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
- Install iNGP dependencies (according to your default CUDA):

```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
If you encounter errors while installing iNGP, it is recommended to check your gcc version. Follow these instructions to change the gcc version within your conda environment. Then return to the repository directory to install iNGP again.
 ```sh
conda install -c conda-forge gxx=9.5.0
cd  $CONDA_PREFIX/lib
ln -s  /usr/lib/x86_64-linux-gnu/libcuda.so ./
cd <your repo directory>
```
