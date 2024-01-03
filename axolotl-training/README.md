For now also need axolotl cloned in this dir.

```
cd axolotl-training
mkdir -p src
git clone  https://github.com/OpenAccess-AI-Collective/axolotl ./src
```

Can run this locally with:

```
sudo cog run accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 --num_machines 1 src/scripts/finetune.py src/examples/openllama-3b/lora.yml
```