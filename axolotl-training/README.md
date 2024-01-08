To run training locally:

```
cog run python train.py --config=config/debug.yaml
```

To run inference locally (assuming you ran the above and have a `training_output.zip` file in the current directory):

```
cog run python predict.py
```

Push to Replicate:

```
cog push r8.im/nateraw/axolotl-trainer-llama-2-7b
```
