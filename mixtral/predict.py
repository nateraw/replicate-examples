import os
import subprocess
import time
from cog import BasePredictor, Input, ConcatenateIterator

from utils import maybe_download_with_pget
from src import Llama


MODEL_ID = "mixtral-8x7b-32kseqlen"
WEIGHTS_URL = "https://weights.replicate.delivery/hf/mixtral-8x7b-32kseqlen"
REMOTE_FILES = [
    "consolidated.00.pth-split00.pth",
    "consolidated.00.pth-split01.pth",
    "consolidated.00.pth-split02.pth",
    "consolidated.00.pth-split03.pth",
    "consolidated.00.pth-split04.pth",
    "consolidated.00.pth-split05.pth",
    "consolidated.00.pth-split06.pth",
    "consolidated.00.pth-split07.pth",
    "consolidated.00.pth-split08.pth",
    "params.json",
    "tokenizer.model",
]
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9


def consolidate_shards(prefix, output_file, num_shards, drop_shards=False):
    if os.path.exists(output_file):
        print("Output file already exists, skipping consolidation")
        return
    print(f"Consolidating {num_shards} shards into {output_file}")
    shard_files = [f"{prefix}-split{i:02d}.pth" for i in range(num_shards)]
    with open(output_file, 'wb') as outfile:
        subprocess.run(['cat'] + shard_files, stdout=outfile)
    if drop_shards:
        for shard in shard_files:
            os.remove(shard)

def load_weights():
    if not os.path.exists("weights-cache/consolidated.00.pth"):
        start = time.time()
        maybe_download_with_pget(
            path="weights-cache",
            remote_path=WEIGHTS_URL,
            remote_filenames=REMOTE_FILES,
        )
        print(f"downloading weights took {time.time() - start:.3f}s...now consolidating shards...")
        consolidate_shards('weights-cache/consolidated.00.pth', 'weights-cache/consolidated.00.pth', 9, drop_shards=False)

class Predictor(BasePredictor):
    def setup(self):
        load_weights()
        # consolidate_shards('weights-cache/consolidated.00.pth', 'weights-cache/consolidated.00.pth', 9)
        self.model = Llama.build(
            ckpt_dir="weights-cache",
            tokenizer_path="weights-cache/tokenizer.model",
            max_seq_len=32768, # 2048 # 512
            max_batch_size=8,
        )

    def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.", default=DEFAULT_TEMPERATURE
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=DEFAULT_TOP_P,
        ),
        # logprobs: bool = Input(
        #     description="Whether to return the log probabilities of the generated tokens.", default=False
        # ),
    ) -> str:
        start = time.time()
        results = self.model.text_completion(
            prompts=[prompt],
            max_gen_len=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            # logprobs=logprobs,
        )
        print(f"generation took {time.time() - start:.3f}s")
        return results[0]["generation"]  # results[0]["logprobs"]


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict(
        "Write a script to download the images for the top 10 posts of all time from /r/pics using the PRAW library.",
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
    )
    print(out)
