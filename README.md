# ScaleDreamer
This is the official repository for ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation.

It is the code base for Prompt-Amortized Text-to-3D Training Scripts, including
- [ ] Compatible with the guidance of
  - [ ] Score Distillation Sampling ([SDS](https://arxiv.org/pdf/2209.14988.pdf))
  - [ ] Variational Score Distillation ([VSD](https://arxiv.org/pdf/2305.16213.pdf))
  - [ ] Classifier Score Distillation ([CSD](https://arxiv.org/pdf/2310.19415.pdf))
  - [ ] Asynchronous Score Distillation (ASD)
- [ ] 3D Generator Architecture
  - [ ] Hyper-iNGP, a reproduction of [ATT3D](https://arxiv.org/pdf/2306.07349.pdf)
  - [ ] 3DConv-net from [CC3D](https://arxiv.org/pdf/2303.12074.pdf)
  - [ ] Triplane-Transformer from [LRM](https://arxiv.org/abs/2311.04400)
- [ ] 2D Diffusion Prior
  - [ ] Stable Diffusion ([SD](https://huggingface.co/stabilityai/stable-diffusion-2-1-base))
  - [ ] MVDream  ([MV](https://arxiv.org/pdf/2308.16512.pdf)) 
- [ ] Text prompt corpus
  - [ ] MG15 from [Magic3D](https://research.nvidia.com/labs/dir/magic3d)
  - [ ] DF415 from [DreamFusion](https://dreamfusion3d.github.io/)
  - [ ] AT2520 from [ATT3D](https://arxiv.org/pdf/2306.07349.pdf)
  - [ ] DL17k from [Instant3D](https://arxiv.org/pdf/2311.08403.pdf)
  - [ ] PP100k from [PickaPic](https://arxiv.org/pdf/2305.01569.pdf)
## Credits
This code is built on the following amazing open-source projects:

- [Threestudio](https://github.com/threestudio-project/threestudio) the versatile text/image-to-3D framework
- [MVDream-Threestudio](https://github.com/bytedance/MVDream-threestudio/tree/main) the implementation of MVDream with Threestudio
