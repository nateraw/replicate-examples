import os
import time
from pathlib import Path
from threading import Thread

import torch
from cog import BasePredictor, ConcatenateIterator, Input

from downloader import Downloader


DEFAULT_MAX_NEW_TOKENS = os.environ.get("DEFAULT_MAX_NEW_TOKENS", 512)
DEFAULT_TEMPERATURE = os.environ.get("DEFAULT_TEMPERATURE", 0.6)
DEFAULT_TOP_P = os.environ.get("DEFAULT_TOP_P", 0.9)
DEFAULT_TOP_K = os.environ.get("DEFAULT_TOP_K", 50)

TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextGenerationPredictor(BasePredictor):
    task = "text-generation"
    hf_model_id = None
    # prompt_template = "{prompt}"
    cache_dir = "./weights-cache"
    load_in_4bit = False
    use_safetensors = True
    generate_kwargs = {}
    local_files_only = True
    gcp_bucket_weights = None
    remote_filenames = None
    torch_dtype = "bf16"
    trust_remote_code = False

    def setup(self):
        downloader = Downloader()
        start = time.time()
        downloader.sync_maybe_download_files(
            self.hf_model_id, self.gcp_bucket_weights, self.remote_filenames
        )
        print(f"downloading weights took {time.time() - start:.3f}s")

        # os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        global TextIteratorStreamer
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            TextIteratorStreamer,
        )

        config = AutoConfig.from_pretrained(
            self.hf_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        # resolve torch dtype from string.
        torch_dtype = TORCH_DTYPE_MAP[self.torch_dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id,
            config=config,
            # quantization_config=bitsandbytes_config if self.load_in_4bit else None,
            torch_dtype=torch_dtype if not self.load_in_4bit else None,
            device_map="auto",
            cache_dir=self.cache_dir,
            use_safetensors=self.use_safetensors,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )

    def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.", default=DEFAULT_TEMPERATURE
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=DEFAULT_TOP_P,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=DEFAULT_TOP_K,
        ),
        prompt_template: str = Input(
            description="The template used to format the prompt before passing it to the model.",
            default="<s>[INST] {prompt} [/INST]",
        )
    ) -> ConcatenateIterator:
        prompt = prompt_template.format(prompt=prompt)
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", add_special_tokens=False, return_token_type_ids=False
        ).to(DEVICE)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for text in streamer:
            yield text

Predictor = TextGenerationPredictor
Predictor.hf_model_id = "mixtral-8x7b-instruct-v0.1"
Predictor.gcp_bucket_weights = "https://weights.replicate.delivery/replicate-weights/mixtral-8x7b-instruct-v0.1"
Predictor.remote_filenames = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    for text in p.predict(
        "Write a hello world FastAPI example",
        max_new_tokens=256,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
    ):
        print(text, end="")