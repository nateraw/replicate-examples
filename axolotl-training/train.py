import os
import subprocess
import time
from argparse import ArgumentParser
from itertools import chain
from zipfile import ZipFile

import psutil
import torch
import yaml
from cog import BaseModel, Path, Input

from utils import maybe_download_with_pget


os.environ["WANDB_ANONYMOUS"] = "must"

OUTPUT_DIR = "./lora-out"
MODEL_WEIGHTS_MAP = {
    "nousresearch/llama-2-7b-hf": {
        "remote_path": "https://weights.replicate.delivery/hf/nousresearch/llama-2-7b-hf/dacdfcde31297e34b19ee0e7532f29586d2c17bc",
        "remote_filenames": [
            "config.json",
            "generation_config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ]
    }
}


class TrainingOutput(BaseModel):
    weights: Path
    # Maybe we separate unnecessary files from weights like this?
    # extra_artifacts: Path


def main(
    config: Path = Input(description="axolotl config file"),
    mixed_precision: str = Input(default="bf16", description="Mixed precision (no,fp16,bf16,fp8)"),
) -> TrainingOutput:
    
    # Check axolotl config to see if base_model is in MODEL_WEIGHTS_MAP.
    # If so, download the weights from GCP. Otherwise, download from the Hugging Face Hub.
    axolotl_cfg = yaml.safe_load(config.read_text())
    base_model = axolotl_cfg["base_model"].lower()
    if base_model in MODEL_WEIGHTS_MAP:
        start = time.time()
        _cfg = MODEL_WEIGHTS_MAP[base_model]
        base_model = f"{base_model}"
        maybe_download_with_pget(
            path=base_model,
            remote_path=_cfg["remote_path"],
            remote_filenames=_cfg["remote_filenames"],
        )
        print(f"downloading base weights took {time.time() - start:.3f}s")

    num_gpus = torch.cuda.device_count()
    multi_gpu = True if num_gpus > 1 else False
    cmd = [
        "accelerate",
        "launch",
        "-m",
    ] + (["--multi_gpu"] if multi_gpu else []) + [
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

    print('-' * 80)
    print(cmd)
    print()
    print(" ".join(cmd))
    print('-' * 80)

    p = None
    try:
        p = subprocess.Popen(cmd, close_fds=False)
        p.wait()
        return_code = p.poll()
        if return_code != 0:
            raise Exception(
                f"Training failed with exit code {return_code}! Check logs for details"
            )

        directory = Path(OUTPUT_DIR)
        def zip_files(output_path, file_paths):
            with ZipFile(output_path, "w") as zip:
                for file_path in file_paths:
                    print(f"Adding file to {output_path}: {file_path}")
                    zip.write(file_path, arcname=file_path.relative_to(directory))

        weights_out_path = "training_output.zip"
        zip_files(weights_out_path, sorted(f for f in directory.glob("*") if f.is_file()))
        return TrainingOutput(weights=Path(weights_out_path))

        ################################################################
        # NOTE: Here's how we'd maybe separate weights from other files
        ################################################################
        # weights_out_path = "training_output.zip"
        # extra_artifacts_out_path = "training_artifacts.zip"
        # zip_files(
        #     weights_out_path,
        #     sorted(
        #         x for x in directory.glob("*")
        #         if x.is_file() and not x.relative_to(directory).name.startswith(('checkpoint'))
        #     )
        # )
        # zip_files(
        #     extra_artifacts_out_path,
        #     sorted(file for pattern in ["runs/**/*", "checkpoint-*/*"] for file in directory.glob(pattern))
        # )
        # return TrainingOutput(weights=Path(weights_out_path), tensorboard_logs=Path(tensorboard_out_path))
        ################################################################

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
    parser.add_argument("--mixed_precision", type=str, default="bf16", help="Mixed Precision config for accelerate, which launches the script. (no,fp16,bf16,fp8)")
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(**vars(args))
