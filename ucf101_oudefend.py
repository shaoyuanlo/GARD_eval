import os
import sys
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from PIL import Image

from spatial_transforms import Compose, Scale, CenterCrop
from model import generate_model
from utils import *

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from armory.data.utils import maybe_download_weights_from_s3
from art.classifiers import PyTorchClassifier
from art.utils import Deprecated


def detector_and_model(detector, model, inputs, spatial_transform):

    #temp_root = 'image.jpg'
    #frame_num = len(inputs[0,0,:,0,0])
    #batch_num = len(inputs[:,0,0,0,0])	
    #for bb in range(batch_num):	
    #    for kk in range(frame_num):	
    #        inputs_frame = torchvision.transforms.ToPILImage()(inputs[bb,:,kk,:,:].cpu())
    #        inputs_frame.save(temp_root)
    #        inputs_frame = Image.open(temp_root).convert('RGB')		
    #        inputs_frame = spatial_transform(inputs_frame)
    #        inputs[bb,:,kk,:,:] = inputs_frame
	#	
    #outputs = detector(inputs)
    #
    #_, type_pred = outputs.topk(1, 1, True)
    #type_pred = type_pred.t().cpu().numpy()
    #
    #outputs = model(inputs, type_pred[:,0])
    outputs = model(inputs)
    
    return outputs


class MyPytorchClassifier(PyTorchClassifier):

    def __init__(self, my_detector, my_model, spatial_transform,
        model=nn.Conv2d(1, 3, 1, bias=False),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 40, 112, 112),
        nb_classes=101,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        channel_index=Deprecated,
        channels_first: bool = True,
        clip_values=(0, 1,),
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing = (0, 1),
        device_type: str = "gpu",
    ):

        # Remove in 1.5.0
        if channel_index == 3:
            channels_first = False
        elif channel_index == 1:
            channels_first = True
        elif channel_index is not Deprecated:
            raise ValueError("Not a proper channel_index. Use channels_first.")
	
        super(MyPytorchClassifier, self).__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channel_index=channel_index,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )

        self.my_detector = my_detector
        self.my_model = my_model
        self.spatial_transform = spatial_transform
		
        cuda_idx = torch.cuda.current_device()
        self._device = torch.device("cuda:{}".format(cuda_idx))
        self.my_detector.to(self._device)
        self.my_model.to(self._device)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.
        :param x: Test set.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        self.my_detector.eval()
        self.my_model.eval()		

        # Apply preprocessing
        x_preprocessed = preprocessing_fn_torch(x)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0]),)
			
            #my_inputs = torch.from_numpy(x_preprocessed[begin:end]).to(self._device)
            my_inputs = x_preprocessed[begin:end].to(self._device)
            #my_inputs = x_preprocessed.to(self._device)
            #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
            #print(my_inputs.shape)

            with torch.no_grad():
                model_outputs = detector_and_model(self.my_detector, self.my_model, my_inputs, self.spatial_transform)

            output = model_outputs[-1]
            results[begin:end] = output.detach().cpu().numpy()

        predictions = results

        return predictions

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.
        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        if not (
            (label is None)
            or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self._nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Apply preprocessing
        x_preprocessed = preprocessing_fn_torch(x)
        x_preprocessed = torch.from_numpy(x_preprocessed).to(self._device)

        # Compute gradients
        if self._layer_idx_gradients < 0:
            x_preprocessed.requires_grad = True

        # Run prediction
        model_outputs = detector_and_model(self.my_detector, self.my_model, x_preprocessed, self.spatial_transform)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_preprocessed

        # Set where to get gradient from
        preds = model_outputs

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self.my_model.zero_grad()
        if label is None:	
            for i in range(self.nb_classes):			
                torch.autograd.backward(
                    preds[:, i], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True,
                )

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True,
            )
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(
                    preds[:, i], torch.tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True,
                )

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        grads = preprocessing_fn_inverse_torch(x, grads)

        return grads

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :return: Array of gradients of the same shape as `x`.
        """
        # Apply preprocessing
        x_preprocessed = preprocessing_fn_torch(x)
        _, y_preprocessed = self._apply_preprocessing(x, y, fit=False)		

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert the inputs to Tensors
        #inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t = x_preprocessed.to(self._device)
        inputs_t.requires_grad_()

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the gradient and return
        inputs_zero = torch.zeros_like(inputs_t)	
        model_outputs = detector_and_model(self.my_detector, self.my_model, inputs_zero + inputs_t, self.spatial_transform)
		
        loss = self._loss(model_outputs, labels_t)

        # Clean gradients
        self.my_model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = inputs_t.grad.cpu().numpy().copy()  # type: ignore
        grads = preprocessing_fn_inverse_torch(x, grads)
        print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')		
        print(x.shape)		
        print(grads.shape)		
        assert grads.shape == x.shape

        return grads

    def loss_gradient_framework(self, x: "torch.Tensor", y: "torch.Tensor", **kwargs) -> "torch.Tensor":
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: Gradients of the same shape as `x`.
        """
        # Check label shape
        if self._reduce_labels:
            y = torch.argmax(y, dim=1)

        # Convert the inputs to Variable
        x = preprocessing_fn_torch(x)		
        x = Variable(x)
        x.requires_grad_()		

        # Compute the gradient and return
        inputs_zero = torch.zeros_like(x)		
        model_outputs = detector_and_model(self.my_detector, self.my_model, inputs_zero + x, self.spatial_transform)
        loss = self._loss(model_outputs, y)

        # Clean gradients
        self.my_model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = x.grad
        assert grads.shape == x.shape  # type: ignore
		
        return grads  # type: ignore


def preprocessing_fn_torch(
    batch: Union[torch.Tensor, np.ndarray],
    consecutive_frames: int = 40,
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
        #logger.warning(f"batch {type(batch)} is not a torch.Tensor. Casting")
        batch = torch.from_numpy(batch).cuda()
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
        #logger.warning("Expected odd example shape (226, 400, 3)")
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
    if len(video) > consecutive_frames:
        # cut trailing frames
        #video = video[: len(video) - (len(video) % consecutive_frames)]
        video = video[: consecutive_frames]

    # Attempts to directly follow MARS approach
    # (frames, height, width, channel) to (frames, channel, height, width)
    video = video.permute(0, 3, 1, 2)
    if standard_shape:  # 240 x 320
        sample_height, sample_width = 112, 149
    else:  # 226 x 400
        video = video[:, :, 1:-1, :]  # crop top/bottom pixels, reduce by 2
        sample_height, sample_width = 112, 200

    video = F.interpolate(video, size=(sample_height, sample_width), mode="bilinear", align_corners=align_corners)

    if standard_shape:
        crop_left = 18  # round((149 - 112)/2.0)
    else:
        crop_left = 40
    video = video[:, :, :, crop_left : crop_left + sample_height]

    if video.max() > 1.0:
        raise ValueError("Video exceeded max after interpolation")
    if video.min() < 0.0:
        raise ValueError("Video under min after interpolation")

    # reshape into stacks of frames
    #video = torch.reshape(video, (-1, consecutive_frames) + video.shape[1:])
    video = torch.unsqueeze(video, 0)

    # transpose to (stacks, channel, stack_frames, height, width)
    video = video.permute(0, 2, 1, 3, 4)
    # video = torch.transpose(video, axes=(0, 4, 1, 2, 3))

    # normalize before changing channel position?
    #video = torch.transpose(video, 1, 4)
    #video = torch.transpose(video, 4, 1)

    return video	


def preprocessing_fn_inverse_torch(x, grads):

    '''
    x.shape = [1, 118, 240, 320, 3]
    grads.shape = [1, 3, 40, 112, 112]
    '''

    consecutive_frames = 40
	
    # (batch, channel, frames, height, width) to (channel, frames, height, width)
    grads = torch.squeeze(grads, 0)
    # resize
    grads = F.interpolate(grads, size=(x.shape[2], x.shape[2]), mode="bilinear", align_corners=align_corners)
    # pad
    p1d = (int((x.shape[3]-x.shape[2])/2), int((x.shape[3]-x.shape[2])/2))
    grads = F.pad(grads, p1d, 'constant', 0)

    # temporal	
    if x.shape[1] <= consecutive_frames:
        grads = grads[:, :x.shape[1], :, :]
    else:
        grads = torch.cat([grads, torch.zeros(x.shape[0], x.shape[1]-consecutive_frames ,x.shape[2], x.shape[3])], dim=1)

    # (channel, frames, height, width) to (frames, height, width, channel)
    grads = grads.permute(1, 2, 3, 0)
    # (batch, frames, height, width, channel) to (frames, height, width, channel)
    grads = torch.unsqueeze(grads, 0)

    return grads	
	
	
def get_my_model(model_kwargs, wrapper_kwargs, weights_file):

    if weights_file:
        pretrain_path = maybe_download_weights_from_s3('JHUM_oudefend.pth')

    my_model = generate_model('resnext_oun')	
    model_data = torch.load(pretrain_path)
    my_model.load_state_dict(model_data['state_dict'])

    spatial_transform = Compose([Scale(112), CenterCrop(112), torchvision.transforms.ToTensor()])

    wrapped_model = MyPytorchClassifier(my_model, my_model, spatial_transform)
	
    return wrapped_model

	