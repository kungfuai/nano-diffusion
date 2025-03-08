## Overview

This folder holds the educational notebooks to learn practical recipes for training image / video generation models under limited resource (under 72 A10 hours of training). Diffusion is a main method. Other adjacent models are also explored. 

## Outline

- 2D points
	- Diffusion (DDPM) [Colab](https://colab.research.google.com/drive/1TIV3GEPLEoZzMBfSm8B5QTeRFNTSVxIu)
	- CFM (conditional flow matching)
- Image generation
	- Image gen (animal faces) using DDPM [Colab](https://colab.research.google.com/drive/1pcFJS5nWDcDEHsig3c4AA8INd52Wjl35)
		- Forward diffusion
		- Denoising model
		- Reverse diffusion (Sampling by Denoising)
		- Training loop
	- Image gen (animal faces) using CFM
- (optional) Path animation for image generation
	- What does forward diffusion look like in a latent space
	- What does reverse diffusion look like
	- How does training change the reverse diffusion paths
- FID: evaluation for image gen [Colab](https://colab.research.google.com/drive/1Qdb8HjVXdN8tvwgoW2rZR86gTPruSrRW)
- Text Conditioning (text2image)
- [Relationship between diffusion and flow matching](https://diffusionflow.github.io/)
- Multi-GPU recipe (8xH100)
	- This may be more cost effective and time saving.
- VAE / tokenizer (to get to high res)
	- SD VAE
	- Cosmos
- Scaling up (e.g. the [50-A100-hours train](https://github.com/apapiu/transformer_latent_diffusion))
- Practical fine-tuning recipes
	- Fine-tune StableDiffusion
	- Fine-tune Flux
- Practical model hosting
	- Framework and hardware choices
- Optional:
	- Conditioning on other modalities (ControlNet)
	- ROPE for long context
	- Video gen
	- Music/speech generation
	- Using representation learning losses to improve training efficiency
	- SANA (nvidia)
	- VAE rotation trick


## References

- [Diffusion course from KAIST, Fall 2024](https://mhsung.github.io/kaist-cs492d-fall-2024/)
- [Distributed training guide from Lambda Labs](https://github.com/LambdaLabsML/distributed-training-guide)
- [Training a diffusion model for protein structure][https://github.com/microsoft/foldingdiff]
- [Diffusion course from MIT, 6.S184][https://diffusion.csail.mit.edu/]