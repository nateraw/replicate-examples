from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetImg2ImgPipeline


CACHE_DIR = "weights-cache"


def resize_for_condition_image(input_image, resolution: int):
    from PIL.Image import LANCZOS

    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=LANCZOS)
    return img


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # torch.backends.cuda.matmul.allow_tf32 = True
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(CACHE_DIR, torch_dtype=torch.float16).to(
            "cuda"
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    def generate_qrcode(self, qr_code_content):
        import qrcode

        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color="white")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)
        return qrcode_image

    # Define the arguments and types the model takes as input
    def predict(
        self,
        prompt: str = Input(description="The prompt to guide QR Code generation."),
        qr_code_content: str = Input(description="The website/content your QR Code will point to."),
        negative_prompt: str = Input(
            description="The negative prompt to guide image generation.",
            default="ugly, disfigured, low quality, blurry, nsfw",
        ),
        num_inference_steps: int = Input(description="Number of diffusion steps", ge=20, le=100, default=40),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=-1),
        batch_size: int = Input(description="Batch size for this prediction", ge=1, le=4, default=1),
        strength: float = Input(
            description="Indicates how much to transform the masked portion of the reference `image`. Must be between 0 and 1.",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        controlnet_conditioning_scale: float = Input(
            description="The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original unet.",
            ge=1.0,
            le=2.0,
            default=1.5,
        ),
    ) -> List[Path]:
        seed = torch.randint(0, 2**32, (1,)).item() if seed == -1 else seed
        qrcode_image = self.generate_qrcode(qr_code_content)
        out = self.pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            image=[qrcode_image] * batch_size,
            control_image=[qrcode_image] * batch_size,
            width=768,
            height=768,
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=torch.Generator().manual_seed(seed),
            strength=float(strength),
            num_inference_steps=num_inference_steps,
        )

        for i, image in enumerate(out.images):
            fname = f"output-{i}.png"
            image.save(fname)
            yield Path(fname)
