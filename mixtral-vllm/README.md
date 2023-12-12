### This example is WIP and not working yet. Here are my notes so far...

Download the safetensors weights for instruct 8x7b model from huggingface hub using the following:

```
pip install hf-transfer huggingface-hub
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Convert to one big file (haven't gotten this to work properly yet, I keep going OOM)

```
python convert_weights.py
```

You can't just `cat shard* > weights.pt` because the weights are structured by layer.
