# ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation

## 🔥 News

- **2024.06.23** Create this repo.

## ⚙️ Dependencies and Installation

<details>
<summary> Follow threestudio to set up the conda environment, or use our provided instructions as below. </summary>
 
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
- Install [iNGP](https://github.com/NVlabs/instant-ngp) and [NerfAcc](https://github.com/nerfstudio-project/nerfacc):

```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/NVlabs/nvdiffrast.git
```
If you encounter errors while installing iNGP, it is recommended to check your gcc version. Follow these instructions to change the gcc version within your conda environment. Then return to the repository directory to install iNGP and NerfAcc ⬆️ again.
 ```sh
conda install -c conda-forge gxx=9.5.0
cd  $CONDA_PREFIX/lib
ln -s  /usr/lib/x86_64-linux-gnu/libcuda.so ./
cd <your repo directory>
```
</details>

<details>
<summary> Download 2D Diffusion Priors. </summary>
 
- Save [SD-v2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and [MVDream](https://mv-dream.github.io/) to the local directory `pretrained`.
 
```
python scripts/download_pretrained_models.py
```
</details>

## 🌈 Prompt-Specific 3D Generation

- ASD with `SD` (Stable Diffusion ). You can change the prompt accordingly.
```
sh scripts/single-prompt-benchmark/asd_sd_nerf.sh
```

- ASD with `MV` (MVDream). You can change the prompt accordingly.
```
sh scripts/single-prompt-benchmark/asd_mv_nerf.sh
```

## 🕹️ Prompt-Amortized 3D Generator Tranining

The following `3D generator` architectures are available: 

| Network | Description | File |
| :-: | :-: | :-: |
| Hyper-iNGP | iNGP with text-conditioned linear layers,adopted from [ATT3D](https://research.nvidia.com/labs/toronto-ai/ATT3D/) | [geometry](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/models/geometry/hyper_iNGP.py), [background](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/extern/triplane_transformer_modules.py)
| Triplane-Transformer | Transformer-based 3D Generator, with [Triplane](https://github.com/NVlabs/eg3d) as the output structure, adopted from [LRM](https://yiconghong.me/LRM/) | [geometry](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/models/geometry/triplane_transformer.py), [architecture](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/models/background/multiprompt_neural_environment_hashgrid_map_background.py)


The following `corpus` datasets are available: [DF415](https://github.com/theEricMa/ScaleDreamer/blob/main/load/dreamfusion_415_prompt_library.json)

| Corpus | Description | File |
| :-: | :-: | :-: |
| MG15 | 15 text pormpts from [Magic3D](https://dreamfusion3d.github.io/) project page | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/magic3d_15_prompt_library.json)
| DF415 | 415 text pormpts from [DreamFusion](https://dreamfusion3d.github.io/) project page | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/dreamfusion_415_prompt_library.json)
| AT2520 | 2520 text pormpts from [ATT3D](https://research.nvidia.com/labs/toronto-ai/ATT3D) experiments | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/att3d_2520_prompt_library.json)
| DL17k | 17k text pormpts from [Instant3D](https://research.nvidia.com/labs/toronto-ai/ATT3D) release | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/att3d_2520_prompt_library.json)
| CP100k | 110k text pormpts from [Cap3D](https://github.com/crockwell/Cap3D) dataset | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/cap3d_100k_prompt_library.json)

Run the following script to start training

- `3DConv-net` with `SD` on `DF415`
 
```sh
sh scripts/multi-prompt-benchmark/asd_sd_3dconv_net_DF415.sh
```
 
## 🚀 Prompt-Amortized 3D Generator Evaluation
Create a directory to save the checkpoints
```sh
mkdir pretrained/3d_checkpoints
```


