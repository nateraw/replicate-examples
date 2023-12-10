# Mixtral example

This example follows along [this example](https://github.com/dzhulgakov/llama-mistral) from [dzhulgakov](https://github.com/dzhulgakov). Go give them a star! :star:

## Local Run

You may need to update `max_seq_len` in `predict.py` to something smaller if you don't have enough compute (256, 512, 1024, etc. instead of 32k).

```
cog run python -i predict.py
```
