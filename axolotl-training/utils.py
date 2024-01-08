import asyncio
import builtins
import contextlib
import os
import subprocess
import typing as tp


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


def download_file(file, local_filename):
    print(f"Downloading {file} to {local_filename}")
    if os.path.exists(local_filename):
        os.remove(local_filename)
    if "/" in local_filename:
        if not os.path.exists(os.path.dirname(local_filename)):
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    command = ["pget", file, local_filename]
    subprocess.check_call(command, close_fds=True)


def check_files_exist(remote_files: list[str], local_path: str) -> list[str]:
    # Get the list of local file names
    local_files = os.listdir(local_path)

    # Check if each remote file exists in the local directory
    missing_files = list(set(remote_files) - set(local_files))

    return missing_files


async def download_file_with_pget(remote_path, dest_path, pget_concurrency="10"):
    # Create the subprocess
    print("Downloading ", remote_path)
    if remote_path.endswith("json"):
        info = "%{filename_effective} took %{time_total}s (%{speed_download} bytes/sec)\n"
        args = ["curl", "-w", info, "-sLo", dest_path, remote_path]
    else:
        args = ["pget", "-c", pget_concurrency, remote_path, dest_path]
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        close_fds=True,
    )

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Print what the subprocess output (if any)
    if stdout:
        print(f"[stdout]\n{stdout.decode()}")
    if stderr:
        print(f"[stderr]\n{stderr.decode()}")


async def download_files_with_pget(remote_path: str, path: str, files: list[str]) -> None:
    download_jobs = "\n".join(f"{remote_path}/{f} {path}/{f}" for f in files)
    print(download_jobs)
    args = ["pget", "multifile", "-", "-f", "--max-conn-per-host", "100"]
    process = await asyncio.create_subprocess_exec(*args, stdin=-1, close_fds=True)
    # Wait for the subprocess to finish
    await process.communicate(download_jobs.encode())


def maybe_download_with_pget(
    path: str,
    remote_path: tp.Optional[str] = None,
    remote_filenames: tp.Optional[list[str]] = None,
):
    """
    Downloads files from remote_path to path if they are not present in path. File paths are constructed
    by concatenating remote_path and remote_filenames. If remote_path is None, files are not downloaded.

    Args:
        path (str): Path to the directory where files should be downloaded
        remote_path (str): Path to the directory where files should be downloaded from
        remote_filenames (List[str]): List of file names to download

    Returns:
        path (str): Path to the directory where files were downloaded

    Example:

        maybe_download_with_pget(
            path="models/roberta-base",
            remote_path="gs://my-bucket/models/roberta-base",
            remote_filenames=["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json"],
        )
    """
    if remote_path:
        remote_path = remote_path.rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            missing_files = remote_filenames or []
        else:
            missing_files = check_files_exist(remote_filenames or [], path)
        get_loop().run_until_complete(download_files_with_pget(remote_path, path, missing_files))

    return path


@contextlib.contextmanager
def delay_prints(REALLY_EAT_MY_PRINT_STATEMENTS: bool = False) -> tp.Iterator[tp.Callable]:
    lines = []

    def delayed_print(*args: tp.Any, **kwargs: tp.Any) -> None:
        lines.append((args, kwargs))

    if REALLY_EAT_MY_PRINT_STATEMENTS:
        builtins.print, _print = delayed_print, builtins.print
    try:
        yield delayed_print
    finally:
        if REALLY_EAT_MY_PRINT_STATEMENTS:
            builtins.print = _print
        for args, kwargs in lines:
            print(*args, **kwargs)

    return delay_prints
