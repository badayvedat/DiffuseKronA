import os
import random

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLPipeline,
)

from cli import get_inference_args
from krona import load_krona_weights
from utils import add_embeddings, read_json_file, replace_tokens


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


args = get_inference_args()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "sdxl-cache", torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = SCHEDULERS[args.scheduler].from_config(pipe.scheduler.config)

special_params = read_json_file(args.special_params_path)
pipe = load_krona_weights(pipe, args.krona_path)
pipe = add_embeddings(args.embedding_path, pipe)

prompt = replace_tokens(args.prompt, special_params)

seed = args.seed
if seed is None:
    seed = random.randint(0, 2**31 - 1)
print(f"Using seed {seed}")

print(f"Prompt: {prompt}")

result = pipe(
    prompt=prompt,
    negative_prompt=args.negative_prompt,
    num_inference_steps=args.num_inference_steps,
    generator=torch.manual_seed(seed),
    num_images_per_prompt=args.num_images,
)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

for idx, image in enumerate(result.images):
    image.save(f"{args.output_path}/{idx}.png")
    print(f"Image saved to {args.output_path}/{idx}.png")
