

def install_megablocks():
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "megablocks"])
        print("Successfully installed megablocks.")
    except subprocess.CalledProcessError:
        print("Failed to install megablocks.")


import asyncio
from typing import AsyncIterator, List, Union
import time
from cog import BasePredictor, Input, ConcatenateIterator
# from cog_hf_template.download_utils import maybe_pget_weights
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from downloader import Downloader


# MODEL_ID = "thebloke/yi-6b-awq"
# PROMPT_TEMPLATE = "{prompt}"
# WEIGHTS_URL = "https://weights.replicate.delivery/hf/thebloke/yi-6b-awq/3eafeffffae6680fa0b084c1cff55aafdb7f0a85"
# REMOTE_FILES = [
#     "LICENSE",
#     "README.md",
#     "config.json",
#     "configuration_yi.py",
#     "generation_config.json",
#     "model.safetensors",
#     "modeling_yi.py",
#     "quant_config.json",
#     "special_tokens_map.json",
#     "tokenization_yi.py",
#     "tokenizer.json",
#     "tokenizer.model",
#     "tokenizer_config.json",
# ]



# MODEL_ID = "mixtral-8x7b-32kseqlen"
MODEL_ID = "mixtral-8x7b-instruct-v0.1"
# WEIGHTS_URL = "https://weights.replicate.delivery/hf/mixtral-8x7b-32kseqlen"
# REMOTE_FILES = [
#     "consolidated.00.pth-split00.pth",
#     "consolidated.00.pth-split01.pth",
#     "consolidated.00.pth-split02.pth",
#     "consolidated.00.pth-split03.pth",
#     "consolidated.00.pth-split04.pth",
#     "consolidated.00.pth-split05.pth",
#     "consolidated.00.pth-split06.pth",
#     "consolidated.00.pth-split07.pth",
#     "consolidated.00.pth-split08.pth",
#     "params.json",
#     "tokenizer.model",
# ]
PROMPT_TEMPLATE = "{prompt}"


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_PRESENCE_PENALTY = 0.0  # 1.15
DEFAULT_FREQUENCY_PENALTY = 0.0  # 0.2


class VLLMPipeline:
    """
    A simplified inference engine that runs inference w/ vLLM
    """

    def __init__(self, *args, **kwargs) -> None:
        args = AsyncEngineArgs(*args, **kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[str]:
        results_generator = self.engine.generate(prompt, sampling_params, 0)
        async for generated_text in results_generator:
            yield generated_text

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: Union[str, List[str]] = None,
        stop_token_ids: List[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        incremental_generation: bool = True,
    ) -> str:
        """
        Given a prompt, runs generation on the language model with vLLM.
        """
        if top_k is None or top_k == 0:
            top_k = -1

        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(
            n=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        gen = self.generate_stream(
            prompt,
            sampling_params,
        )

        generation_length = 0
        while True:
            try:
                request_output = loop.run_until_complete(gen.__anext__())
                assert len(request_output.outputs) == 1
                generated_text = request_output.outputs[0].text
                if incremental_generation:
                    yield generated_text[generation_length:]
                else:
                    yield generated_text
                generation_length = len(generated_text)
            except StopAsyncIteration:
                break


class Predictor(BasePredictor):
    def setup(self):
        # maybe_pget_weights(
        #     path=MODEL_ID,
        #     remote_path=WEIGHTS_URL,
        #     remote_filenames=REMOTE_FILES,
        # )
        # Example usage
        install_megablocks()
        # downloader = Downloader()
        # start = time.time()
        # downloader.sync_maybe_download_files(
        #     MODEL_ID, WEIGHTS_URL, REMOTE_FILES
        # )
        # print(f"downloading weights took {time.time() - start:.3f}s")
        self.llm = VLLMPipeline(
            MODEL_ID,
            # quantization="awq",
            dtype="auto",
            tensor_parallel_size=2,
            trust_remote_code=True,
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
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=DEFAULT_TOP_K,
        ),
        presence_penalty: float = Input(
            description="Presence penalty",
            default=DEFAULT_PRESENCE_PENALTY,
        ),
        frequency_penalty: float = Input(
            description="Frequency penalty",
            default=DEFAULT_FREQUENCY_PENALTY,
        ),
        prompt_template: str = Input(
            description="The template used to format the prompt. The input prompt is inserted into the template using the `{prompt}` placeholder.",
            default=PROMPT_TEMPLATE,
        )
    ) -> ConcatenateIterator[str]:
        start = time.time()
        generate = self.llm(
            prompt=prompt_template.format(prompt=prompt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        for text in generate:
            yield text
        print(f"generation took {time.time() - start:.3f}s")



# def main():
if __name__ == "__main__":
    p = Predictor()
    p.setup()
    for text in p.predict(
        "Here are the top 10 best all-time movie quotes:\n\n1. ",
        512,
        0.8,
        0.95,
        50,
        1.0,
        0.2,
        PROMPT_TEMPLATE,
    ):
        print(text, end="")
