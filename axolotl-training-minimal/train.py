import os
import subprocess
import time
from argparse import ArgumentParser
from zipfile import ZipFile

import psutil
import torch
import yaml
from cog import BaseModel, Input, Path

from zipfile import ZipFile


def zip_files(directory, output_path, file_paths):
    with ZipFile(output_path, "w") as zip:
        for file_path in file_paths:
            print(f"Adding file to {output_path}: {file_path}")
            zip.write(file_path, arcname=file_path.relative_to(directory))


# Enables anonymous logging to wandb
os.environ["HF_HOME"] = "./hf-cache"
os.environ["WANDB_ANONYMOUS"] = "must" if not os.environ.get("WANDB_API_KEY") else "allow"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "true"
# where the adapter weights will be saved
OUTPUT_DIR = "./lora-out"


class TrainingOutput(BaseModel):
    weights: Path


def main(
    config: Path = Input(description="axolotl config file"),
    mixed_precision: str = Input(default="bf16", description="Mixed precision (no,fp16,bf16,fp8)"),
) -> TrainingOutput:
    axolotl_cfg = yaml.safe_load(config.read_text())
    print(f"----- axolotl_cfg -----\n{yaml.dump(axolotl_cfg)}\n-----------------------\n")
    base_model = axolotl_cfg["base_model"].lower()
    num_gpus = torch.cuda.device_count()
    multi_gpu = True if num_gpus > 1 else False
    cmd = (
        [
            "accelerate",
            "launch",
            "-m",
        ]
        + (["--multi_gpu"] if multi_gpu else [])
        + [
            f"--mixed_precision={mixed_precision}",
            f"--num_processes={num_gpus}",
            "--num_machines=1",
            "--dynamo_backend=no",
            "axolotl.cli.train",
            f"{config}",
            f"--base_model={base_model}",
            f"--output_dir={OUTPUT_DIR}",
            "--save_total_limit=1",
        ]
    )

    print("-" * 80)
    print(cmd)
    print("-" * 80)

    p = None
    try:
        p = subprocess.Popen(cmd, close_fds=False)
        p.wait()
        return_code = p.poll()
        if return_code != 0:
            raise Exception(f"Training failed with exit code {return_code}! Check logs for details")
        directory = Path(OUTPUT_DIR)
        weights_out_path = Path("training_output.zip")
        zip_files(
            directory,
            weights_out_path,
            sorted(f for f in directory.glob("*") if f.is_file())
        )
        return TrainingOutput(weights=weights_out_path)
    finally:
        if p and p.poll() is None:
            top = psutil.Process(p.pid)
            children = top.children(recursive=True)
            for process in children + [top]:
                process.terminate()
            _, alive = psutil.wait_procs(children + [top], timeout=5)
            if alive:
                for process in alive:
                    print(f"process {process.pid} survived termination")
            else:
                print("terminated all processes successfully")


def parse_args(args: str = None):
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="axolotl config file")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Mixed Precision config for accelerate, which launches the script. (no,fp16,bf16,fp8)",
    )
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(**vars(args))
