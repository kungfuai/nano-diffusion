from diffusers import DDPMPipeline
from pathlib import Path
import numpy as np
from cleanfid import fid

model_id = "google/ddpm-cifar10-32"
n=100
# n = 50000

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

# run pipeline in inference (sample random noise and denoise)
ddpm.to("cuda")

images = []
batch_size = 100
for _ in range(n // batch_size):
    images += [i for i in ddpm(batch_size=batch_size).images]

# Save images

gen_path = Path(f"results/sampled_img_{n}")
gen_path.mkdir(exist_ok=True, parents=True)

for i, img in enumerate(images[:]):
    img_np = np.array(img)
    np.save(gen_path / f"{i}.npy", img_np)

score = fid.compute_fid(
            str(gen_path),
            dataset_name="cifar10",
            dataset_res=32,
            device="cuda",
            mode="clean",
            batch_size=2,
            num_workers=0,
        )

print("FID score:", score)  # 156.41708381872482
