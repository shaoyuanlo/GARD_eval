  
"""
Model contributed by: MITRE Corporation
Adapted from: https://github.com/craston/MARS
"""
import logging
from typing import Union, Optional, Tuple

from art.classifiers import PyTorchClassifier
import numpy as np
from PIL import Image
import torch
from torch import optim

from model import generate_model

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing_fn_torch(
    batch: Union[torch.Tensor, np.ndarray],
    consecutive_frames: int = 40,
    scale_first: bool = True,
    align_corners: bool = False,
):
    """
    inputs - batch of videos each with shape (frames, height, width, channel)
    outputs - batch of videos each with shape (n_stack, channel, stack_frames, new_height, new_width)
        frames = n_stack * stack_frames (after padding)
        new_height = new_width = 112
    consecutive_frames - number of consecutive frames (stack_frames)
    After resizing, a center crop is performed to make the image square
    This is a differentiable alternative to MARS' PIL-based preprocessing.
        There are some
    """
    if not isinstance(batch, torch.Tensor):
        logger.warning(f"batch {type(batch)} is not a torch.Tensor. Casting")
        batch = torch.from_numpy(batch).to(DEVICE)
        # raise ValueError(f"batch {type(batch)} is not a torch.Tensor")
    if batch.dtype != torch.float32:
        raise ValueError(f"batch {batch.dtype} should be torch.float32")
    if batch.shape[0] != 1:
        raise ValueError(f"Batch size {batch.shape[0]} != 1")
    video = batch[0]

    if video.ndim != 4:
        raise ValueError(
            f"video dims {video.ndim} != 4 (frames, height, width, channel)"
        )
    if video.shape[0] < 1:
        raise ValueError("video must have at least one frame")
    if tuple(video.shape[1:]) == (240, 320, 3):
        standard_shape = True
    elif tuple(video.shape[1:]) == (226, 400, 3):
        logger.warning("Expected odd example shape (226, 400, 3)")
        standard_shape = False
    else:
        raise ValueError(f"frame shape {tuple(video.shape[1:])} not recognized")
    if video.max() > 1.0 or video.min() < 0.0:
        raise ValueError("input should be float32 in [0, 1] range")
    if not isinstance(consecutive_frames, int):
        raise ValueError(f"consecutive_frames {consecutive_frames} must be an int")
    if consecutive_frames < 1:
        raise ValueError(f"consecutive_frames {consecutive_frames} must be positive")

    # Select a integer multiple of consecutive frames
    while len(video) < consecutive_frames:
        # cyclic pad if insufficient for a single stack
        video = torch.cat([video, video[: consecutive_frames - len(video)]])
    if len(video) % consecutive_frames != 0:
        # cut trailing frames
        video = video[: len(video) - (len(video) % consecutive_frames)]

    if scale_first:
        # Attempts to directly follow MARS approach
        # (frames, height, width, channel) to (frames, channel, height, width)
        video = video.permute(0, 3, 1, 2)
        if standard_shape:  # 240 x 320
            sample_height, sample_width = 112, 149
        else:  # 226 x 400
            video = video[:, :, 1:-1, :]  # crop top/bottom pixels, reduce by 2
            sample_height, sample_width = 112, 200

        video = torch.nn.functional.interpolate(
            video,
            size=(sample_height, sample_width),
            mode="bilinear",
            align_corners=align_corners,
        )

        if standard_shape:
            crop_left = 18  # round((149 - 112)/2.0)
        else:
            crop_left = 40
        video = video[:, :, :, crop_left : crop_left + sample_height]

    else:
        # More efficient, but not MARS approach
        # Center crop
        sample_size = 112
        if standard_shape:
            crop_width = 40
        else:
            video = video[:, 1:-1, :, :]
            crop_width = 88
        video = video[:, :, crop_width:-crop_width, :]

        # Downsample to (112, 112) frame size
        # (frames, height, width, channel) to (frames, channel, height, width)
        video = video.permute(0, 3, 1, 2)
        video = torch.nn.functional.interpolate(
            video,
            size=(sample_size, sample_size),
            mode="bilinear",
            align_corners=align_corners,
        )

    if video.max() > 1.0:
        raise ValueError("Video exceeded max after interpolation")
    if video.min() < 0.0:
        raise ValueError("Video under min after interpolation")

    # reshape into stacks of frames
    video = torch.reshape(video, (-1, consecutive_frames) + video.shape[1:])

    # transpose to (stacks, channel, stack_frames, height, width)
    video = video.permute(0, 2, 1, 3, 4)

    return video


def make_model(
    model_status: str = "ucf101_trained", weights_path: Optional[str] = None
) -> Tuple[torch.nn.DataParallel, optim.SGD]:

    model = generate_model('resnext_3bn_comb')
    checkpoint = torch.load(weights_path, map_location=DEVICE)

    model_dict = model.state_dict()	
    for k, v in checkpoint['state_dict'].items():
        k = k.replace('.module', '')
        model_dict[k] = v
    model.load_state_dict(model_dict)

    return model


class OuterModel(torch.nn.Module):
    def __init__(
        self, weights_path: Optional[str], max_frames: int = 0, **model_kwargs,
    ):
        """
        Max frames is the maximum number of input frames.
            If max_frames == 0, no clipping is done
            Else if max_frames > 0, frames are clipped to that number.
            This can be helpful for smaller memory cards.
        """
        super().__init__()
        max_frames = int(max_frames)
        if max_frames < 0:
            raise ValueError(f"max_frames {max_frames} cannot be negative")
        self.max_frames = max_frames
        self.model = make_model(
            weights_path=weights_path, **model_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_frames:
            x = x[:, : self.max_frames]

        if self.training:
            # Use preprocessing_fn_numpy in dataset preprocessing
            return self.model(x)
        else:
            x = preprocessing_fn_torch(x)
            stack_outputs = self.model(x)
            output = stack_outputs.mean(axis=0, keepdims=True)

        return output


def get_my_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = OuterModel(weights_path=weights_path, **model_kwargs)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(None, 240, 320, 3),
        nb_classes=101,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
	
	