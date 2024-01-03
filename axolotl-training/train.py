from argparse import ArgumentParser
import subprocess
from zipfile import ZipFile
import torch
from cog import BaseModel, Path, Input
import psutil


class TrainingOutput(BaseModel):
    weights: Path


def main(
    mixed_precision: str = Input(default="bf16", description="Mixed precision (no,fp16,bf16,fp8)"),
    micro_batch_size: int = Input(default=2, description="Micro batch size"),
    gradient_accumulation_steps: int = Input(default=4, description="Gradient accumulation steps"),
    val_set_size: float = 0.1,
    evals_per_epoch: int = 1,
) -> TrainingOutput:
    output_dir = "./lora-out"
    num_gpus = torch.cuda.device_count()
    multi_gpu = True if num_gpus > 1 else False
    multi_gpu_str = "--multi_gpu" if multi_gpu else ""
    cmd = [
        "accelerate",
        "launch",
    ] + ([multi_gpu_str] if multi_gpu else []) + [
        f"--mixed_precision={mixed_precision}",
        f"--num_processes={num_gpus}",
        "--num_machines=1",
        "src/scripts/finetune.py",
        "src/examples/llama-2/lora.yml",
        f"--output_dir={output_dir}",
        f"--micro_batch_size={micro_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--val_set_size={val_set_size}",
        f"--evals_per_epoch={evals_per_epoch}"
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
        out_path = "training_output.zip"

        directory = Path(output_dir)
        with ZipFile(out_path, "w") as zip:
            for file_path in directory.rglob("*"):
                print(file_path)
                zip.write(file_path, arcname=file_path.relative_to(directory))

        return TrainingOutput(weights=Path(out_path))
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
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(**vars(args))
