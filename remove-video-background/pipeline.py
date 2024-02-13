import requests
from io import BytesIO

import torch
import numpy as np
import torch.nn.functional as F
from pytorchvideo.transforms import ApplyTransformToKey, Div255, Normalize, Permute
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV
from torchvision.transforms import Compose, Resize

from briarmbg import BriaRMBG


def download_video_bytes(url):
    """
    Downloads video from the given URL and returns its bytes.

    Parameters:
    - url (str): The URL of the video to download.

    Returns:
    - bytes: The bytes of the downloaded video.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors.

        video_bytes = response.content
        return video_bytes
    except requests.RequestException as e:
        print(f"Error downloading the video: {e}")
        return None


def postprocess_video(video_tensor: torch.Tensor, im_size: list) -> np.ndarray:
    # Assuming video_tensor shape is (T, C, H, W)
    video_tensor = F.interpolate(video_tensor, size=im_size, mode='bilinear', align_corners=False)
    ma = torch.max(video_tensor)
    mi = torch.min(video_tensor)
    video_tensor = (video_tensor - mi) / (ma - mi)
    video_array = (video_tensor * 255).permute(0, 2, 3, 1).squeeze(-1).cpu().numpy().astype(np.uint8)
    return video_array


def apply_mask_to_video_with_background_color(video: torch.Tensor, mask: np.ndarray, background_color: tuple) -> torch.Tensor:
    # Ensure mask normalization and shape adjustment for broadcasting
    mask_tensor = torch.from_numpy(mask).to(video.device, dtype=video.dtype) / 255.0  # Normalize to 0-1
    mask_tensor = mask_tensor.unsqueeze(1)  # Shape: (T, 1, H, W)

    # Adjust video to (T, C, H, W) to apply mask
    video_permuted = video.permute(1, 0, 2, 3)  # Now shape is (T, C, H, W)

    # Correct the creation of the background tensor
    # Need to create a background color tensor that matches the video shape
    background_color_tensor = torch.tensor(background_color, device=video.device, dtype=video.dtype).view(1, 3, 1, 1)  # Shape: (1, C, 1, 1)
    # Expand background color tensor to match the video's T and spatial dimensions (T, C, H, W)
    background_tensor = background_color_tensor.expand(video_permuted.size(0), 3, video_permuted.size(2), video_permuted.size(3))

    # Calculate the inverse mask for background application
    inverse_mask = 1 - mask_tensor

    # Apply the mask to the video and the inverse mask to the background color
    masked_video = video_permuted * mask_tensor + background_tensor * inverse_mask

    # Permute back to original video shape (C, T, H, W)
    masked_video = masked_video.permute(1, 0, 2, 3)

    return masked_video


# import torch
# from pytorchvideo.transforms import ApplyTransformToKey, Compose, Resize, Div255, Normalize, Permute
# from pytorchvideo.data.encoded_video import EncodedVideo

class Pipeline:
    def __init__(self, model_name="briaai/RMBG-1.4", device=None, input_size=(1024, 1024)):
        # Initialize device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.net = BriaRMBG.from_pretrained(model_name)
        self.net.to(self.device)

        # Set model input size
        self.model_input_size = input_size

        # Define preprocessing transformations
        self.preprocess_transform = ApplyTransformToKey(
            key='video',
            transform=Compose([
                Resize(self.model_input_size),
                Div255(),
                Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                Permute((1, 0, 2, 3)),
            ])
        )

    def __call__(self, video_filepath, chunk_duration=1.0):
        if "http" in str(video_filepath):
            bpayload = download_video_bytes(video_filepath)
            vid = EncodedVideoPyAV(BytesIO(bpayload))
        else:
            # Load video
            vid = EncodedVideoPyAV.from_path(video_filepath)

        full_duration = float(vid.duration)

        # Calculate total number of chunks
        num_chunks = int(full_duration // chunk_duration) + (1 if full_duration % chunk_duration != 0 else 0)

        fps = float(vid._container.streams.video[0].average_rate)
        audio_fps = (
            None if not vid._has_audio else vid._container.streams.audio[0].sample_rate
        )
        audio_codec = (
            None if not vid._has_audio else vid._container.streams.audio[0].codec.name
        )

        # TODO - Consider improving how we feed data to the model to keep GPU hot
        processed_video_clips = []
        processed_audio_clips = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, full_duration)
            print(i, start_time, "-", end_time)
            # Get video clip
            clip = vid.get_clip(start_time, end_time)
            if clip['video'] is None:
                break
            original_clip = clip['video'].clone()
            if clip['audio'] is not None:
                original_audio = clip['audio'].clone()  # Extract audio clip
            original_shape = list(clip['video'].shape[2:])

            # Preprocess video
            out = self.preprocess_transform(clip)

            # Predict with model
            with torch.no_grad():
                result = self.net(out['video'].to(self.device))[0][0].cpu()

            # Postprocess the result
            result_video = postprocess_video(result, original_shape)

            # Apply mask to original video
            video_out = apply_mask_to_video_with_background_color(original_clip, result_video, (0, 255, 0))

            processed_video_clips.append(video_out)
            if clip['audio'] is not None:
                processed_audio_clips.append(original_audio)

        # Concatenate processed clips
        full_processed_video = torch.cat(processed_video_clips, dim=1)  # Assuming dim=1 is the time dimension

        # Concatenate processed audio
        # Assuming original_audio is a tensor and dim=1 is the time dimension for audio as well
        if processed_audio_clips:
            full_processed_audio = torch.cat(processed_audio_clips, dim=0) if processed_audio_clips else None
        else:
            full_processed_audio = None

        return {'video': full_processed_video, 'audio': full_processed_audio, "fps": fps, "audio_fps": audio_fps, "audio_codec": audio_codec}


if __name__ == '__main__':
    from torchvision.io import write_video
    input_file_or_url = "https://huggingface.co/spaces/nateraw/animegan-v2-for-videos/resolve/main/obama.webm"
    out_filepath = "obama_bg_removed.mp4"

    pipe = Pipeline()
    out = pipe(input_file_or_url, chunk_duration=1.0)
    fps, audio_fps, audio_codec = out['fps'], out['audio_fps'], out['audio_codec']
    if out['audio'] is not None:
        write_video(
            out_filepath,
            out['video'].permute(1, 2, 3, 0),
            audio_array=out['audio'].unsqueeze(0),
            audio_codec="aac",
            audio_fps=audio_fps,
            fps=fps
        )
    else:
        write_video(
            out_filepath,
            out['video'].permute(1, 2, 3, 0),
            fps=fps
        )
