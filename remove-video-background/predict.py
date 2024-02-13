# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import tempfile
from typing import Union
# set HF_HOME before importing any HF libraries
os.environ["HF_HOME"] = "./hf-cache"
# enable fast downloads using hf-transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from cog import BasePredictor, Input, Path
from torchvision.io import write_video

from pipeline import Pipeline


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe = Pipeline()

    def predict(
        self,
        video: Path = Input(description="Input video"),
    ) -> Path:
        out = self.pipe(str(video), chunk_duration=1.5)
        fps, audio_fps = out['fps'], out['audio_fps']

        out_path = Path(tempfile.mkdtemp()) / "out.mp4"
        if out['audio'] is not None:
            write_video(
                str(out_path),
                out['video'].permute(1, 2, 3, 0),  # C, T, H, W -> T, H, W, C
                audio_array=out['audio'].unsqueeze(0),
                audio_codec="aac",
                audio_fps=audio_fps,
                fps=fps,
                video_codec='h264',
                options={'crf': '10'}
            )
        else:
            write_video(
                str(out_path),
                out['video'].permute(1, 2, 3, 0),  # C, T, H, W -> T, H, W, C
                fps=fps,
                video_codec='h264',
                options={'crf': '10'}
            )
        return out_path


if __name__ == '__main__':
    predictor = Predictor()
    predictor.setup()

    input_file_or_url = "https://replicate.delivery/pbxt/KObDFVxLp3hoAWCPsvxdVjHJLtN23IN1cHIv0XDyAjnOg0II/obama_1_trimmed.mp4"
    out_path = predictor.predict(video=input_file_or_url)

    ###########################################################################
    # To run the test below, first download a video and remove its audio track
    # wget -nc https://huggingface.co/spaces/nateraw/animegan-v2-for-videos/resolve/main/obama.webm
    # ffmpeg -y -i obama.webm -c:v copy -an obama_noaudio.webm
    ###########################################################################
    # input_file_or_url = "obama_noaudio.webm"
    # out_path = predictor.predict(video=input_file_or_url)
