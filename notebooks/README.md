## Overview

This folder holds the educational notebooks to learn practical recipes for training image / video generation models under limited resource (under 72 A10 hours of training). Diffusion is a main method. Other adjacent models are also explored. 

## Outline

- 2D points
	- Diffusion (DDPM) [Colab](https://colab.research.google.com/drive/1TIV3GEPLEoZzMBfSm8B5QTeRFNTSVxIu)
	- CFM (conditional flow matching)
- Image gen (CIFAR10 or animal faces) using DDPM [Colab](https://colab.research.google.com/drive/1pcFJS5nWDcDEHsig3c4AA8INd52Wjl35)
    - Forward diffusion
    - Denoising model
    - Reverse diffusion (Sampling by Denoising)
    - Training loop
- (inspirational) Path animation for image generation
	- What does forward diffusion look like in a latent space
	- What does reverse diffusion look like
	- How does training change the reverse diffusion paths
- Image gen (CIFAR10 or animal faces) using CFM
- FID: evaluation for image gen [Colab](https://colab.research.google.com/drive/1Qdb8HjVXdN8tvwgoW2rZR86gTPruSrRW)
- Conditioning on Text (text2image)
- Multi-GPU recipe (8xH100)
	- This may be more cost effective and time saving.
- VAE / tokenizer (to get to high res)
- Optional:
	- Conditioning on other modalities (ControlNet)
	- Video gen


## References

- [Diffusion course from KAIST, Fall 2024](https://mhsung.github.io/kaist-cs492d-fall-2024/)
- [Distributed training guide from Lambda Labs](https://github.com/LambdaLabsML/distributed-training-guide)