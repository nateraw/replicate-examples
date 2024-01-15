## Running Locally

```
cog run python train.py --config config/debug.yaml
```

This will produce a `training_output.zip`` file in the current directory, which you can then use to run inference with the predictor:

```
cog run python predict.py
```

If running locally, you can log directly to your own weights and biases profile with:

```
export WANDB_API_KEY=YOUR-WANDB-API-KEY

cog run -e WANDB_API_KEY=$WANDB_API_KEY python train.py --config=config/tinyllama_debug.yaml
```