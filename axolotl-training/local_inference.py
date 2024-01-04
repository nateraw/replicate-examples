from datasets import load_from_disk
from pathlib import Path

import torch
from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline


#############################################################################
# Config
#############################################################################
# base model
# model_id="nousresearch/llama-2-7b-hf"
# adapter model
peft_model_id="lora-out"
# If left to none, we just grab first hash dir within last_run_prepared.
# Ex: "last_run_prepared/<hash>"
# Feel free to replace with exact path of your choice.
data_dir=None
#############################################################################

data_dir = data_dir or next(Path('last_run_prepared').glob("*"))
ds = load_from_disk(data_dir)

peft_config = PeftConfig.from_pretrained(peft_model_id)
base_model_id = peft_config.base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
decoded_ex = tokenizer.decode(ds[0]['input_ids'])
print(f"----- Decoded example -----")
print(decoded_ex)
print()

PROMPT_TEMPLATE = """\
### Instruction:
{prompt}

### Response:
 """

print(f"----- Loading Base Model -----")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("\ndone")

print(f"----- Loading Adapter Model -----")
model.load_adapter(peft_model_id)
print("\ndone")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=peft_model_id,
)

print(f"----- Generating -----")
out = pipe(
    PROMPT_TEMPLATE.format(prompt="Give three tips for staying healthy."),
    return_full_text=False,
    do_sample=False,
    max_new_tokens=256
)[0]['generated_text']
print(out)
print('----------------------')
