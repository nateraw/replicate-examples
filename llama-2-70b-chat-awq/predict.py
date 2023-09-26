import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from vllm import LLM, SamplingParams
import torch
from cog import BasePredictor, Input

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_PRESENCE_PENALTY = 1.0


PROMPT_TEMPLATE = """\
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{message}[/INST]"""

class Predictor(BasePredictor):

    def setup(self):
        self.llm = LLM(
            model="TheBloke/Llama-2-70B-chat-AWQ",
            quantization="awq",
            dtype="float16"
        )

    def predict(
        self,
        message: str,
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
    ) -> str:
        prompts = [PROMPT_TEMPLATE.format(message=message)]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty
        )
        start = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        print(f"\nGenerated {len(outputs[0].outputs[0].token_ids)} tokens in {time.time() - start} seconds.")
        return outputs[0].outputs[0].text


if __name__ == '__main__':
    p = Predictor()
    p.setup()
    out = p.predict("Write me an itinerary for my dog's birthday party.", 1024, 0.8, 0.95, 50, 1.0)
    print(out)
