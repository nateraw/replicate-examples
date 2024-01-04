import os
import time
from threading import Thread
from typing import Optional, Any
import zipfile

import torch
from cog import BasePredictor, Path, ConcatenateIterator, Input

from utils import maybe_download_with_pget, download_file
import functools

DEFAULT_MAX_NEW_TOKENS = os.environ.get("DEFAULT_MAX_NEW_TOKENS", 512)
DEFAULT_TEMPERATURE = os.environ.get("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.environ.get("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.environ.get("DEFAULT_TOP_K", 50)

TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_ID = "nousresearch/llama-2-7b-hf"  # "tinyllama/tinyllama-1.1b-intermediate-step-1431k-3t"
PROMPT_TEMPLATE = """\
<s> ### System:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""


class Predictor(BasePredictor):
    task = "text-generation"
    base_model_id = BASE_MODEL_ID
    cache_dir = "./hf-cache"
    use_safetensors = None
    local_files_only = False
    gcp_bucket_weights = None
    remote_filenames = None
    torch_dtype = "bf16"
    trust_remote_code = False

    def setup(self, weights: Optional[Path] = None):
        print("Starting setup")
        
        if self.gcp_bucket_weights:
            start = time.time()
            maybe_download_with_pget(
                self.hf_model_id, self.gcp_bucket_weights, self.remote_filenames
            )
            print(f"downloading weights took {time.time() - start:.3f}s")

        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        global TextIteratorStreamer
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TextIteratorStreamer,
        )
        # resolve torch dtype from string.
        torch_dtype = TORCH_DTYPE_MAP[self.torch_dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=self.cache_dir,
            use_safetensors=self.use_safetensors,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
            use_flash_attention_2=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )

        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        if weights:
            if "http" in str(weights):  # weights are in the cloud
                print("Downloading peft weights")
                st = time.time()
                download_file(str(weights), "training_output.zip")
                print(f"Downloaded peft weights in {time.time() - st:.3f}")
                remote_weights_or_path = "training_output.zip"
            else:
                # zipfile accepts either a file-like or path-like object
                remote_weights_or_path = weights

            peft_model_dir = "peft_model"

            st = time.time()
            with zipfile.ZipFile(remote_weights_or_path, "r") as zip_ref:
                zip_ref.extractall(peft_model_dir)
            print(f"Unzipped peft weights in {time.time() - st:.3f}")
            st = time.time()
            self.model.load_adapter(peft_model_dir)
            print(f"Initialized peft model in {time.time() - st:.3f}")
            print(f"----- LORA WEIGHTS LOADED FROM: {peft_model_dir} -----")
        else:
            print("----- NOT USING ADAPTER MODEL -----")

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
            description="The template used to format the prompt before passing it to the model. For no template, you can set this to `{prompt}`.",
            default=PROMPT_TEMPLATE,
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


if __name__ == "__main__":
    p = Predictor()
    p.setup(weights=Path("training_output.zip"))
    # p.setup(weights=Path("https://replicate.delivery/pbxt/ZTCf8Ezzhy0BNCHEtfzRVEQhyT2HQB8MdLYRUYFfRsBD22RkA/training_output.zip"))
    for text in p.predict(
        PROMPT_TEMPLATE.format(prompt="Give three tips for becoming a better writer."),
        max_new_tokens=1024,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
        prompt_template=PROMPT_TEMPLATE,
    ):
        print(text, end="")