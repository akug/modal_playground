
import scripts.control_utils as cu
import torch
from PIL import Image
from diffusers import DiffusionPipeline

def download_sdxl():
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

#download_sdxl()

path_to_config = 'ControlNet-XS/configs/inference/sdxl/sdxl_encD_canny_48m.yaml'
model = cu.create_model(path_to_config)

model = model.to('cuda')

image_path = 'PATH/TO/IMAGES/Shoe.png'

canny_high_th = 250
canny_low_th = 100
size = 768
num_samples=2

image = cu.get_image(image_path, size=size)
edges = cu.get_canny_edges(image, low_th=canny_low_th, high_th=canny_high_th)

samples, controls = cu.get_sdxl_sample(
    guidance=edges,
    ddim_steps=10,
    num_samples=num_samples,
    model=model,
    shape=[4, size // 8, size // 8],
    control_scale=0.95,
    prompt='cinematic, shoe in the streets, made from meat, photorealistic shoe, highly detailed',
    n_prompt='lowres, bad anatomy, worst quality, low quality',
)