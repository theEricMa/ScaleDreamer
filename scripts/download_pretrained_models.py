from diffusers import DiffusionPipeline

repo_id = "stabilityai/stable-diffusion-2-1-base"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
pipeline.save_pretrained("./pretrained/stable-diffusion-2-1-base")

import os
cmd = "wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt?download=true -O ./pretrained/sd-v2.1-base-4view.pt"
os.system(cmd)
