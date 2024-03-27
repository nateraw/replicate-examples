import os

os.environ["HF_HOME"] = "./hf-cache"
import asyncio
from pathlib import Path
from typing import AsyncIterator, List, Union
from uuid import uuid4
import time
from cog import BasePredictor, Input, ConcatenateIterator
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
import torch

from utils import maybe_download_with_pget, delay_prints

import json

cfg = json.loads(Path("config.json").read_text())
print(f"CFG: {json.dumps(cfg, indent=2, sort_keys=True)}")

MODEL_ID = cfg["model_id"]
WEIGHTS_URL = cfg["weights_url"]
REMOTE_FILES = cfg["remote_filenames"]
PROMPT_TEMPLATE = cfg["prompt_template"]
TRUST_REMOTE_CODE = cfg.get("trust_remote_code", False)

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_PRESENCE_PENALTY = 0.0  # 1.15
DEFAULT_FREQUENCY_PENALTY = 0.0  # 0.2


class VLLMPipeline:
    """
    A simplified inference engine that runs inference w/ vLLM
    """

    def __init__(self, *args, **kwargs) -> None:
        args = AsyncEngineArgs(*args, **kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer.tokenizer

    async def generate_stream(self, prompt: str, sampling_params: SamplingParams) -> AsyncIterator[str]:
        request_id = uuid4().hex
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        async for generated_text in results_generator:
            yield generated_text

    async def __call__(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: Union[str, List[str]] = None,
        stop_token_ids: List[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        incremental_generation: bool = True,
    ) -> str:
        """
        Given a prompt, runs generation on the language model with vLLM.
        """
        if top_k is None or top_k == 0:
            top_k = -1

        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(
            n=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        gen = self.generate_stream(
            prompt,
            sampling_params,
        )

        generation_length = 0
        async for request_output in gen:
            assert len(request_output.outputs) == 1
            generated_text = request_output.outputs[0].text
            if incremental_generation:
                yield generated_text[generation_length:]
            else:
                yield generated_text
            generation_length = len(generated_text)


class Predictor(BasePredictor):
    async def setup(self):
        start = time.time()
        maybe_download_with_pget(MODEL_ID, WEIGHTS_URL, REMOTE_FILES)
        print(f"downloading weights took {time.time() - start:.3f}s")
        self.llm = VLLMPipeline(
            MODEL_ID,
            dtype="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=TRUST_REMOTE_CODE,
            # max_model_len=256,
            **{"quantization": "awq"} if "awq" in MODEL_ID else {},
        )

    async def predict(
        self,
        # prompt: str,
        question: str,
        table_metadata: str,
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
        presence_penalty: float = Input(
            description="Presence penalty",
            default=DEFAULT_PRESENCE_PENALTY,
        ),
        frequency_penalty: float = Input(
            description="Frequency penalty",
            default=DEFAULT_FREQUENCY_PENALTY,
        ),
        prompt_template: str = Input(
            description="The template used to format the prompt. The input prompt is inserted into the template using the `{prompt}` placeholder.",
            default=PROMPT_TEMPLATE,
        ),
    ) -> ConcatenateIterator[str]:
        with delay_prints(REALLY_EAT_MY_PRINT_STATEMENTS=True):
            start = time.time()
            generate = self.llm(
                # prompt=prompt_template.format(prompt=prompt),
                prompt=prompt_template.format(question=question, table_metadata=table_metadata),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            async for text in generate:
                yield text
            print(f"\ngeneration took {time.time() - start:.3f}s")


table_metadata = """\
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson 
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region 
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred 
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id 
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id"""

question = (
    "Do we get more sales from customers in New York compared to customers in San Francisco? Give me the total sales for each city, and the difference between the two.",
)


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    result = ""
    for text in p.predict(
        question,
        table_metadata,
        512,
        0.01,
        0.95,
        50,
        1.0,
        0.2,
        PROMPT_TEMPLATE,
    ):
        result += text
        print(text, end="")
