import os
from threading import Thread
from typing import Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path

from utils import download_and_unzip_weights

# Set HF_HOME before importing transformers
CACHE_DIR = "./hf-cache"
os.environ["HF_HOME"] = CACHE_DIR
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from peft import PeftConfig


BASE_MODEL_ID = "nousresearch/llama-2-7b-hf"
PROMPT_TEMPLATE = "{prompt}"
PEFT_MODEL_DIR = "peft_model"

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("Starting setup")
        base_model_id = BASE_MODEL_ID

        if weights:
            print(f"Weights: {weights}")
            download_and_unzip_weights(weights, PEFT_MODEL_DIR)
            config = PeftConfig.from_pretrained(PEFT_MODEL_DIR)
            base_model_id = config.base_model_name_or_path
            print(f"Overriding default Base model id {BASE_MODEL_ID} with: {base_model_id}")
        else:
            print("----- NOT USING ADAPTER MODEL -----")


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if weights:
            print(f"loading adapter from {PEFT_MODEL_DIR}")
            self.model.load_adapter(PEFT_MODEL_DIR)

    def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=512,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.", default=0.7
        ),
        do_sample: bool = Input(
            description="Whether or not to use sampling; otherwise use greedy decoding.", default=True
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=0.95,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=50,
        ),
        prompt_template: str = Input(
            description="The template used to format the prompt before passing it to the model. For no template, you can set this to `{prompt}`.",
            default=PROMPT_TEMPLATE,
        ),
    ) -> ConcatenateIterator:
        prompt = prompt_template.format(prompt=prompt)
        print(f"=== Formatted Prompt ===\n{prompt}\n{'=' * 24}\n")
        inputs = self.tokenizer([prompt], return_tensors="pt", return_token_type_ids=False).to(self.device)
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


_prompt_template = """\
### System:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the input from English to Hinglish

### Input:
{prompt}

### Response:
"""


if __name__ == "__main__":
    p = Predictor()
    p.setup(weights="training_output.zip")
    for text in p.predict(
        "What time is the game tomorrow?",
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        top_p=0.95,
        top_k=50,
        prompt_template=_prompt_template,
    ):
        print(text, end="")
