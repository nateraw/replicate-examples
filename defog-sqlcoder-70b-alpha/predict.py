import os
from threading import Thread
from typing import Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path


# Set HF_HOME before importing transformers
CACHE_DIR = "./hf-cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from peft import PeftConfig


MODEL_ID = "defog/sqlcoder-70b-alpha"
PROMPT_TEMPLATE = """\
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{table_metadata}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("Starting setup")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def predict(
        self,
        question: str = Input(description="The question to answer with a SQL query."),
        table_metadata: str = Input(
            description="The database schema to use when generating the SQL query."
        ),
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
        prompt = prompt_template.format(question=question, table_metadata=table_metadata)
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


_example_schema_str = """\
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

if __name__ == "__main__":
    p = Predictor()
    p.setup(weights="training_output.zip")
    for text in p.predict(
        question="Do we get more sales from customers in New York compared to customers in San Francisco? Give me the total sales for each city, and the difference between the two.",
        table_metadata=_example_schema_str,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        top_p=0.95,
        top_k=50,
        prompt_template=PROMPT_TEMPLATE,
    ):
        print(text, end="")
