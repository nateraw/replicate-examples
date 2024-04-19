# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

from datetime import datetime

# We need to set cache before any imports, which is why this is up here.
MODEL_PATH = "./hf-and-torch-cache"
os.environ["HF_HOME"] = MODEL_PATH
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TORCH_HOME"] = MODEL_PATH

from cog import BasePredictor, Input, Path
import torch

import torchaudio
import subprocess
import typing as tp
import numpy as np

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

BATCH_SIZE = 4

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MusicGen.get_pretrained('nateraw/musicgen-songstarter-v0.2', device='cuda')

    def predict(
        self,
        prompt: str = Input(
            description="A description of the music you want to generate.", default=None
        ),
        input_audio: Path = Input(
            description="An audio file that will influence the generated music. If `continuation` is `True`, the generated music will be a continuation of the audio file. Otherwise, the generated music will mimic the audio file's melody.",
            default=None,
        ),
        duration: int = Input(
            description="Duration of the generated audio in seconds.", default=8
        ),
        continuation: bool = Input(
            description="If `True`, generated music will continue from `input_audio`. Otherwise, generated music will mimic `input_audio`'s melody.",
            default=False,
        ),
        continuation_start: int = Input(
            description="Start time of the audio file to use for continuation.",
            default=0,
            ge=0,
        ),
        continuation_end: int = Input(
            description="End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip.",
            default=None,
            ge=0,
        ),
        normalization_strategy: str = Input(
            description="Strategy for normalizing audio.",
            default="loudness",
            choices=["loudness", "clip", "peak", "rms"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If None or -1, a random seed will be used.",
            default=None,
        ),
    ) -> tp.List[Path]:
        set_generation_params = lambda duration: self.model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        if not input_audio:
            set_generation_params(duration)
            wav, tokens = self.model.generate([prompt] * BATCH_SIZE, progress=True, return_tokens=True)
        else:
            input_audio, sr = torchaudio.load(input_audio)
            input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

            continuation_start = 0 if not continuation_start else continuation_start
            if continuation_end is None or continuation_end == -1:
                continuation_end = input_audio.shape[2] / sr

            if continuation_start > continuation_end:
                raise ValueError(
                    "`continuation_start` must be less than or equal to `continuation_end`"
                )

            input_audio_wavform = input_audio[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            input_audio_wavform = input_audio_wavform.repeat(BATCH_SIZE, 1, 1)
            input_audio_duration = input_audio_wavform.shape[-1] / sr

            if continuation:
                set_generation_params(duration)# + input_audio_duration)
                print("Continuation wavform shape!", input_audio_wavform.shape)
                wav, tokens = self.model.generate_continuation(
                    prompt=input_audio_wavform,
                    prompt_sample_rate=sr,
                    descriptions=[prompt] * BATCH_SIZE,
                    progress=True,
                    return_tokens=True
                )
            else:
                print("Melody wavform shape!", input_audio_wavform.shape)
                set_generation_params(duration)
                wav, tokens = self.model.generate_with_chroma(
                    [prompt] * BATCH_SIZE, input_audio_wavform, sr, progress=True, return_tokens=True
                )

        dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        paths = []
        for i in range(BATCH_SIZE):
            wav_path_stem = f"{dt_str}_out_{i:03d}"
            wav_path = f"{wav_path_stem}.wav"
            paths.append(wav_path)
            audio_write(
                wav_path_stem,
                wav[i].cpu(),
                self.model.sample_rate,
                strategy=normalization_strategy,
            )

            if output_format == "mp3":
                mp3_path = f"{wav_path_stem}.mp3"
                if os.path.isfile(mp3_path):
                    os.remove(mp3_path)
                subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
                os.remove(wav_path)
                path = mp3_path
            else:
                path = wav_path

        return Path(paths[0]), Path(paths[1]), Path(paths[2]), Path(paths[3])

    def _preprocess_audio(
        audio_path, model: MusicGen, duration: tp.Optional[int] = None
    ):

        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)

        # Calculate duration in seconds if not provided
        if duration is None:
            duration = wav.shape[1] / model.sample_rate

        # Check if duration is more than 30 seconds
        if duration > 30:
            raise ValueError("Duration cannot be more than 30 seconds")

        end_sample = int(model.sample_rate * duration)
        wav = wav[:, :end_sample]

        assert wav.shape[0] == 1
        assert wav.shape[1] == model.sample_rate * duration

        wav = wav.cuda()
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            gen_audio = model.compression_model.encode(wav)

        codes, scale = gen_audio

        assert scale is None

        return codes


# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    # text to audio
    # out = predictor.predict(
    #     "rap, synth, electronic, A minor, 140 bpm",
    #     None,
    #     8,
    #     True,
    #     0,
    #     None,
    #     "loudness",
    #     250,
    #     0,
    #     1.0,
    #     3,
    #     "wav",
    #     42,
    # )
    # melody conditioned text to audio
    # out = predictor.predict(
    #     "hip hop, piano, keys, melody, choir, trap, songstarters, a minor, 160 bpm",
    #     "kalhonaho.wav",
    #     24,
    #     False,
    #     0,
    #     None,
    #     "loudness",
    #     250,
    #     0,
    #     1.0,
    #     3,
    #     "wav",
    #     42,
    # )
    # out = predictor.predict(
    #     "hip hop, piano, keys, melody, choir, trap, songstarters, a minor, 160 bpm",
    #     "kalhonaho.wav",
    #     24,
    #     False,
    #     0,
    #     None,
    #     "loudness",
    #     250,
    #     0,
    #     1.0,
    #     3,
    #     "wav",
    #     42,
    # )
    # out = predictor.predict(
    #     "hip hop, piano, keys, melody, choir, trap, songstarters, a minor, 160 bpm",
    #     "kalhonaho.wav",
    #     24,
    #     False,
    #     0,
    #     None,
    #     "loudness",
    #     250,
    #     0,
    #     1.0,
    #     3,
    #     "wav",
    #     42,
    # )
    # out = predictor.predict(
    #     "hip hop, piano, keys, melody, choir, trap, songstarters, a minor, 160 bpm",
    #     "kalhonaho.wav",
    #     24,
    #     False,
    #     0,
    #     None,
    #     "loudness",
    #     250,
    #     0,
    #     1.0,
    #     3,
    #     "wav",
    #     42,
    # )
    prompt = "synth, hip hop, arp, melody, plucks, trap, future bass, a minor, 160 bpm"
    for i in range(4):
        out = predictor.predict(
            prompt,
            "kalhonaho.wav",
            24,
            False,
            0,
            None,
            "loudness",
            250,
            0,
            1.0,
            3,
            "wav",
            None,
        )
# # continuation w/ text to audio
    # out = predictor.predict(
    #     "rap, synth, electronic, A minor, 140 bpm",
    #     "kalhonaho.wav",
    #     8,
    #     True,
    #     0,
    #     4,
    #     "loudness",
    #     250,
    #     0,
    #     1.0,
    #     3,
    #     "wav",
    #     42,
    # )
    # print(out)
