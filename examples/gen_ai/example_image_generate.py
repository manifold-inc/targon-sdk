import io
import random
from pathlib import Path
from typing import Optional

import targon

MINUTES = 60
CACHE_DIR = "/cache"

image = (
    targon.Image.debian_slim()
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

app = targon.App("example-text-to-image", image=image)

# Import packages that are only available in the remote environment
with image.imports():
    import diffusers
    import torch

MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

@app.function(resource="h200-small",timeout=10 * MINUTES)
def generate_image(prompt: str, batch_size: int = 4, seed: Optional[int] = None) -> list[bytes]:
    
    pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Generate images
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    print(f"🎲 Seeding RNG with {seed}")
    torch.manual_seed(seed)
    
    print(f"🎨 Generating {batch_size} image(s) from prompt: '{prompt}'")
    images = pipe(
        prompt,
        num_images_per_prompt=batch_size,
        num_inference_steps=4,
        guidance_scale=0.0,
        max_sequence_length=512,
    ).images

    # Convert to bytes
    image_output = []
    for image in images:
        with io.BytesIO() as buf:
            image.save(buf, format="PNG")
            image_output.append(buf.getvalue())
    
    torch.cuda.empty_cache()
    print("✨ Generation complete!")
    return image_output



# targon run example-text-to-image --message "A prince riding on a pony"
@app.local_entrypoint()
async def main(
    prompt: str = "A prince riding on a pony",
    batch_size: int = 4,
    seed: Optional[int] = None,
):
    images = await generate_image.remote(prompt, batch_size, seed)
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)
    for i, img_bytes in enumerate(images):
        path = output_dir / f"image_{i}.png"
        with open(path, "wb") as f:
            f.write(img_bytes)
    return "Image saved to output_images!"
