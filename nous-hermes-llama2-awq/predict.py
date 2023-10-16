import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from vllm import LLM, SamplingParams
import torch
from cog import BasePredictor, Input, ConcatenateIterator
import typing as t


MODEL_ID = "TheBloke/Nous-Hermes-Llama2-AWQ"
PROMPT_TEMPLATE = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_PRESENCE_PENALTY = 0.0  # 1.15
DEFAULT_FREQUENCY_PENALTY = 0.0  # 0.2


def vllm_generate_iterator(
    self, prompt: str, /, *, echo: bool = False, stop: str  = None, stop_token_ids: t.List[int] = None, sampling_params=None, **attrs: t.Any
) -> t.Iterator[t.Dict[str, t.Any]]:
    request_id: str = attrs.pop('request_id', None)
    if request_id is None: raise ValueError('request_id must not be None.')
    if stop_token_ids is None: stop_token_ids = []
    stop_token_ids.append(self.tokenizer.eos_token_id)
    stop_ = set()
    if isinstance(stop, str) and stop != '': stop_.add(stop)
    elif isinstance(stop, list) and stop != []: stop_.update(stop)
    for tid in stop_token_ids:
        if tid: stop_.add(self.tokenizer.decode(tid))

    # if self.config['temperature'] <= 1e-5: top_p = 1.0
    # else: top_p = self.config['top_p']
    # config = self.config.model_construct_env(stop=list(stop_), top_p=top_p, **attrs)
    self.add_request(request_id=request_id, prompt=prompt, sampling_params=sampling_params)

    token_cache = []
    print_len = 0

    while self.has_unfinished_requests():
        for request_output in self.step():
            # Add the new tokens to the cache
            for output in request_output.outputs:
                text = output.text
                yield {'text': text, 'error_code': 0, 'num_tokens': len(output.token_ids)}

            if request_output.finished: break


class Predictor(BasePredictor):

    def setup(self):
        self.llm = LLM(
            model=MODEL_ID,
            quantization="awq",
            dtype="float16"
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
    ) -> ConcatenateIterator:
        prompts = [
            (
                prompt_template.format(prompt=prompt),
                SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty
                )
            )
        ]
        start = time.time()
        while True:
            if prompts:
                prompt, sampling_params = prompts.pop(0)
                gen = vllm_generate_iterator(self.llm.llm_engine, prompt, echo=False, stop=None, stop_token_ids=None, sampling_params=sampling_params, request_id=0)
                last = ""
                for _, x in enumerate(gen):
                    if x['text'] == "":
                        continue
                    yield x['text'][len(last):]
                    last = x["text"]
                    num_tokens = x["num_tokens"]
                print(f"\nGenerated {num_tokens} tokens in {time.time() - start} seconds.")

                if not (self.llm.llm_engine.has_unfinished_requests() or prompts):
                    break


if __name__ == '__main__':
    import sys
    p = Predictor()
    p.setup()
    gen = p.predict(
        "Write me an itinerary for my dog's birthday party.",
        512,
        0.8,
        0.95,
        50,
        1.0,
        0.2,
        PROMPT_TEMPLATE,
    )
    for out in gen:
        print(out, end="")
        sys.stdout.flush()