from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
out_path = "mixtral-8x7b-instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained(out_path, safe_serialization=True, max_shard_size="100GB")
tokenizer.save_pretrained(out_path)