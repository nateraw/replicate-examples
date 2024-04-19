# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
from functools import partial

import librosa
import soundfile as sf
from cog import BasePredictor, Input, Path

from pitch_correction_utils import autotune, closest_pitch, aclosest_pitch_from_scale


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        audio_file: Path = Input(description="Audio input file"),
        scale: str = Input(
            description="Strategy for normalizing audio.",
            default="closest",
            choices=["closest", "A:maj", "A:min", "Bb:maj", "Bb:min", "B:maj", "B:min", "C:maj", "C:min", "Db:maj", "Db:min", "D:maj", "D:min", "Eb:maj", "Eb:min", "E:maj", "E:min", "F:maj", "F:min", "Gb:maj", "Gb:min", "G:maj", "G:min", "Ab:maj", "Ab:min"],
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        filepath = Path(audio_file)

        # Load the audio file.
        y, sr = librosa.load(str(filepath), sr=None, mono=False)

        # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
        if y.ndim > 1:
            y = y[0, :]

        # Pick the pitch adjustment strategy according to the arguments.
        correction_function = closest_pitch if scale == 'closest' else \
            partial(aclosest_pitch_from_scale, scale=scale)

        # Perform the auto-tuning.
        pitch_corrected_y = autotune(y, sr, correction_function, plot=False)

        # Write the corrected audio to an output file.
        filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
        sf.write(str(filepath), pitch_corrected_y, sr)

        if output_format == "mp3":
            mp3_path = f"{filepath.stem}.mp3"
            if os.path.isfile(mp3_path):
                os.remove(mp3_path)
            subprocess.call(["ffmpeg", "-i", filepath, mp3_path])
            os.remove(filepath)
            path = mp3_path
        else:
            path = filepath

        return Path(path)


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict(
        audio_file="./nate_is_singing_Gb_minor.wav",
        scale="closest",
        output_format="mp3",
    )
