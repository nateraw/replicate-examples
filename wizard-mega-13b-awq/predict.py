import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from vllm import LLM, SamplingParams
import torch
from cog import BasePredictor, Input


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
DEFAULT_PRESENCE_PENALTY = 1.15


PROMPT_TEMPLATE = """### Instruction:
{message}

### Response:
"""

class Predictor(BasePredictor):

    def setup(self):
        self.llm = LLM(
            model="TheBloke/wizard-mega-13B-AWQ",
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
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs[0].outputs[0].text


if __name__ == '__main__':
    p = Predictor()
    p.setup()
    out = p.predict("Write me an itinerary for my dog's birthday party.", 512, 0.8, 0.95, 50, 1.0)
    print(out)
