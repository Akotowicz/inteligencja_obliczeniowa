from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, UNet2DConditionModel
import torch
torch.cuda.empty_cache()

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())

# load pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# load finetuned model
unet_id = "mhdang/dpo-sd1.5-text2image-v1"
unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
pipe.unet = unet
pipe = pipe.to("cuda")

prompt = "Photorealistic mountain view with forest and deer"
image = pipe(prompt, guidance_scale=7.5).images[0].resize((512,512))

image.save("zad4.2.png")
# # show image
# import matplotlib.pyplot as plt
# plt.imshow(image)
