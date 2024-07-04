# ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation

### [Paper](https://arxiv.org/pdf/2407.02040) | [Project Page](https://sites.google.com/view/scaledreamer-release/)

## 🔥 News

- **2024.07.02** ScaleDreamer is accepted by ECCV 2024
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

- ASD with `SD` (Stable Diffusion). Feel free to change the prompt accordingly.
```
sh scripts/single-prompt-benchmark/asd_sd_nerf.sh
```

- ASD with `MV` (MVDream). Feel free to change the prompt accordingly.
```
sh scripts/single-prompt-benchmark/asd_mv_nerf.sh
```

## 🚀 Prompt-Amortized 3D Generator Tranining

The following `3D generator` architectures are available: 

| Network | Description | File |
| :-: | :-: | :-: |
| Hyper-iNGP | iNGP with text-conditioned linear layers,adopted from [ATT3D](https://research.nvidia.com/labs/toronto-ai/ATT3D/) | [geometry](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/models/geometry/hyper_iNGP.py), [background](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/extern/triplane_transformer_modules.py)
| 3DConv-net | A StyleGAN generator that outputs voxels with 3D convolution, code adopted from [CC3D](https://github.com/sherwinbahmani/cc3d/blob/master/training/networks_3d.py) | [geometry](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/models/geometry/stylegan_3dconv_net.py), [architecture](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/extern/stylegan_3dconv_modules.py)
| Triplane-Transformer | Transformer-based 3D Generator, with [Triplane](https://github.com/NVlabs/eg3d) as the output structure, adopted from [LRM](https://yiconghong.me/LRM/) | [geometry](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/models/geometry/triplane_transformer.py), [architecture](https://github.com/theEricMa/ScaleDreamer/blob/main/custom/amortized/models/background/multiprompt_neural_environment_hashgrid_map_background.py)


The following `corpus` datasets are available: 

| Corpus | Description | File |
| :-: | :-: | :-: |
| MG15 | 15 text pormpts from [Magic3D](https://dreamfusion3d.github.io/) project page | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/magic3d_15_prompt_library.json)
| DF415 | 415 text pormpts from [DreamFusion](https://dreamfusion3d.github.io/) project page | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/dreamfusion_415_prompt_library.json)
| AT2520 | 2520 text pormpts from [ATT3D](https://research.nvidia.com/labs/toronto-ai/ATT3D) experiments | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/att3d_2520_prompt_library.json)
| DL17k | 17k text pormpts from [Instant3D](https://research.nvidia.com/labs/toronto-ai/ATT3D) release | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/att3d_2520_prompt_library.json)
| CP100k | 110k text pormpts from [Cap3D](https://github.com/crockwell/Cap3D) dataset | [json](https://github.com/theEricMa/ScaleDreamer/blob/main/load/cap3d_100k_prompt_library.json)

Run the following script to start training

- `Hyper-iNGP` with `SD` on `MG15`

```sh
sh scripts/multi-prompt-benchmark/asd_sd_hyper_iNGP_MG15.sh
```

- `3DConv-net` with `SD` on `DF415`
 
```sh
sh scripts/multi-prompt-benchmark/asd_sd_3dconv_net_DF415.sh
```

- `3DConv-net` with `SD` on `AT2520`

```sh
sh scripts/multi-prompt-benchmark/asd_sd_3dconv_net_AT2520.sh
```


- `Triplane-Transformer` with `MV` on `DL17k`

```sh
sh scripts/multi-prompt-benchmark/asd_mv_triplane_transformer_DL17k.sh
```

- `3DConv-net` with `SD` on `CP100k`

```sh
scripts/multi-prompt-benchmark/asd_sd_3dconv_net_CP100k.sh
```

## 📷 Prompt-Amortized 3D Generator Evaluation
Create a directory to save the checkpoints
```sh
mkdir pretrained/3d_checkpoints
```

The checkpoints of the ⬆️ experiments are available. Save the corresponding `.pth` file to `3d_checkpoint`, then run the scripts as below.

- `Hyper-iNGP` with `SD` on `MG15`. The ckpt in [Google Drive](https://drive.google.com/file/d/1gjj1QEeu4OR3VATw7g7-zFIJpmg2A5kR/view?usp=sharing)

```sh
sh scripts/multi_prompts_benchmark_evaluation/asd_sd_hyper_iNGP_MG15.sh
```

- `3DConv-net` with `SD` on `DF415`. The ckpt in [Google Drive](https://drive.google.com/file/d/1elAPZlYNRRaA4jH9eb5xCJH5VrgofzbJ/view?usp=sharing)
 
```sh
sh scripts/multi_prompts_benchmark_evaluation/asd_sd_3dconv_net_DF415.sh
```

- `3DConv-net` with `SD` on `AT2520`. The ckpt in [Google Drive](https://drive.google.com/file/d/1hJcdWqr6aB6D5J2ixioRZPsyJn5dmiVp/view?usp=sharing)

```sh
sh scripts/multi_prompts_benchmark_evaluation/asd_sd_3dconv_net_AT2520.sh
```


- `Triplane-Transformer` with `MV` on `DL17k`. The ckpt in [Google Drive](https://drive.google.com/file/d/155N80gHsbXTdsRVAc07lGrbzlHpdNMns/view?usp=sharing)

```sh
sh scripts/multi_prompts_benchmark_evaluation/asd_mv_triplane_transformer_DL17k.sh
```

- `3DConv-net` with `SD` on `CP100k`. The ckpt in [Google Drive](https://drive.google.com/file/d/1SGPS6jWFmC9FahdlIk3SZOn2t0JPHHNE/view?usp=sharing)

```sh
sh scripts/multi_prompts_benchmark_evaluation/asd_sd_3dconv_net_CP100k.sh
```

The rendered images and videos are saved in `outputs/<experiment_name>/save/<num_iter>` directory. Compute the metrics with CLIP via

```
python evaluation/CLIP/evaluation_amortized.py --result_dir <video_dir>
```

## 🕹️ Create Your Own Modules

### 3D Generator

1. Place the code in `custom/amortized/models/geometry`, check out the other code in that directory for reference.
2. Update your <name_of_file> in `custom/amortized/models/geometry/__init__.py`
3. Create your own config file, type in your registered module name in the `system.geometry_type` argument, check out the other code in the `configs/multi-prompt_benchmark` directory for reference.
   
### 2D Diffusion Guidance

1. Put your code in `threestudio/models/guidance`, take a look at the other code in that directory or other [guidance](https://github.com/threestudio-project/threestudio/tree/main/threestudio/models/guidance) for reference.
2. Update your <name_of_file> in  `threestudio/models/guidance/__init__.py`
3. Create your own config file, type in your registered module name in the `system.guidance_type` argument, take a look at the other code in the `configs/multi-prompt_benchmark` directory for reference.


### Text corpus

1. Create a JSON file that lists the training, validation, and test text prompts in the `load` directory 
2. Enter the name of this JSON file into the `system.prompt_processor.prompt_library` argument to set up the corpus, take other commands in the `scripts` directory for reference

You can also add your modules for `data`, `renderer`, `prompt_processor`, etc.

## 🙏 Acknowledgement

- [threestudio](https://github.com/threestudio-project/threestudio), a clean and extensible codebase for text-to-3D.
- [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio), the implementation of MVDream for text-to-3D.
- [OpenLRM](https://github.com/3DTopia/OpenLRM), the implementation of [LRM](https://yiconghong.me/LRM/). We develop the 3D generator of Triplane-Transformer on top of it.
- [Cap3D](https://cap3d-um.github.io/), which provides the text caption of [Objaverse](https://objaverse.allenai.org/). We develop the corpus of CP100k on top of it.


