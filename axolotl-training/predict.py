import os
import time
import zipfile
from threading import Thread
from typing import Optional

import asyncio
import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from train import MODEL_WEIGHTS_MAP
from utils import download_file, maybe_download_with_pget


# Set HF_HOME before importing transformers
CACHE_DIR = "./hf-cache"
os.environ["HF_HOME"] = CACHE_DIR
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

BASE_MODEL_ID = "nousresearch/llama-2-7b-hf"
PROMPT_TEMPLATE = """\
### System:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the input from English to Hinglish

### Input:
{prompt}

### Response:
"""

# Lookup weights from the available models pushed to GCP.
# If they're not available there, they will be downloaded from Hugging Face.
# Note that downloading of the base weights effects cold-start time significantly.
REMOTE_PATH = MODEL_WEIGHTS_MAP.get(BASE_MODEL_ID.lower(), {}).get("remote_path")
REMOTE_FILENAMES = MODEL_WEIGHTS_MAP.get(BASE_MODEL_ID.lower(), {}).get("remote_filenames")

# More options / defaults
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SAFETENSORS = None
LOCAL_FILES_ONLY = False
TORCH_DTYPE = "auto"
TRUST_REMOTE_CODE = False


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("Starting setup")

        if REMOTE_PATH:
            start = time.time()
            maybe_download_with_pget(BASE_MODEL_ID, REMOTE_PATH, REMOTE_FILENAMES)
            print(f"downloading weights took {time.time() - start:.3f}s")

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
            cache_dir=CACHE_DIR,
            use_safetensors=USE_SAFETENSORS,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=TRUST_REMOTE_CODE,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            cache_dir=CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

        if weights:
            print(f"Weights: {weights}")
            if "http" in str(weights):  # weights are in the cloud
                print("Downloading peft weights")
                st = time.time()
                download_file(str(weights), "training_output.zip")
                print(f"Downloaded peft weights in {time.time() - st:.3f}")
                print(os.listdir("."))
                print()
                remote_weights_or_path = "training_output.zip"
            else:
                # zipfile accepts either a file-like or path-like object
                remote_weights_or_path = weights

            peft_model_dir = "peft_model"

            st = time.time()
            with zipfile.ZipFile(remote_weights_or_path, "r") as zip_ref:
                zip_ref.extractall(peft_model_dir)
            print(f"Unzipped peft weights in {time.time() - st:.3f}")
            print(os.listdir('peft_model'))
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
        do_sample: bool = Input(
            description="Whether or not to use sampling; otherwise use greedy decoding.", default=True
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
        ),
    ) -> ConcatenateIterator:
        prompt = prompt_template.format(prompt=prompt)
        print(f"=== Formatted Prompt ===\n{prompt}\n========= END ==========")
        inputs = self.tokenizer([prompt], return_tensors="pt", return_token_type_ids=False).to(DEVICE)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            num_beams=1,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        for text in streamer:
            yield text


if __name__ == "__main__":
    p = Predictor()
    p.setup(weights="https://replicate.delivery/pbxt/nZV9MXQmZ25nGdd11kkcaoZPmWQNeKpbdydvr2sESAdQsLFJA/training_output.zip")
    for text in p.predict(
        "What time is the game tomorrow?",
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=0.0,
        do_sample=False,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
        prompt_template=PROMPT_TEMPLATE,
    ):
        print(text, end="")
