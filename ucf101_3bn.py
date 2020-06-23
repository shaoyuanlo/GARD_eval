import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import torchvision
from torch.autograd import Variable

from spatial_transforms import Compose, Scale, CenterCrop
from model import generate_model
from utils import *

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING


def detector_and_model(detector, model, inputs, spatial_transform):

    temp_root = '/home/sylo/SegNet/armory_mine/video_classification/image.jpg'
    frame_num = len(inputs[0,0,:,0,0])
    batch_num = len(inputs[:,0,0,0,0])	
    for bb in range(batch_num):	
        for kk in range(frame_num):	
            inputs_frame = torchvision.transforms.ToPILImage()(inputs[bb,:,kk,:,:].cpu())
            inputs_frame.save(temp_root)
            inputs_frame = Image.open(temp_root).convert('RGB')		
            inputs_frame = spatial_transform(inputs_frame)
            inputs[bb,:,kk,:,:] = inputs_frame
		
    outputs = detector(inputs)
    
    _, type_pred = outputs.topk(1, 1, True)
    type_pred = type_pred.t().cpu().numpy()
    
    outputs = model(inputs, type_pred[:,0])
    
    return outputs


class MyPytorchClassifier():

    def __init__(self, detector, model, spatial_transform):		
        super(MyPytorchClassifier, self).__init__()

        self.detector = detector
        self.model = model
        self.spatial_transform = spatial_transform
		
        self.nb_classes = 101
        self._loss = nn.CrossEntropyLoss()		
		
        cuda_idx = torch.cuda.current_device()
        self._device = torch.device("cuda:{}".format(cuda_idx))
        self.detector.to(self._device)
        self.model.to(self._device)

    def predict(self, x: np.ndarray, batch_size: int=8, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.
        :param x: Test set.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        self.detector.eval()
        self.model.eval()		

        # Apply preprocessing
        x_preprocessed = x

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0]),)
			
            my_inputs = torch.from_numpy(x_preprocessed[begin:end]).to(self._device)

            with torch.no_grad():
                model_outputs = detector_and_model(self.detector, self.model, my_inputs, self.spatial_transform)

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
        x_preprocessed = x
        x_preprocessed = torch.from_numpy(x_preprocessed).to(self._device)

        # Compute gradients
        if self._layer_idx_gradients < 0:
            x_preprocessed.requires_grad = True

        # Run prediction
        model_outputs = detector_and_model(self.detector, self.model, x_preprocessed, self.spatial_transform)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_preprocessed

        # Set where to get gradient from
        preds = model_outputs[-1]

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self.model.zero_grad()
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
        grads = self._apply_preprocessing_gradient(x, grads)

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
        x_preprocessed, y_preprocessed = x, y

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the gradient and return
        model_outputs = detector_and_model(self.detector, self.model, inputs_t, self.spatial_transform)
        loss = self._loss(model_outputs[-1], labels_t)

        # Clean gradients
        self.model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = inputs_t.grad.cpu().numpy().copy()  # type: ignore
        grads = self._apply_preprocessing_gradient(x, grads)
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
        x = Variable(x, requires_grad=True)

        # Compute the gradient and return
        model_outputs = detector_and_model(self.detector, self.model, x, self.spatial_transform)
        loss = self._loss(model_outputs[-1], y)

        # Clean gradients
        self.model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = x.grad
        assert grads.shape == x.shape  # type: ignore

        return grads  # type: ignore
	
def get_my_model(model_kwargs, wrapper_kwargs, weights_file):

    detector_path = 'model_weight/save_4.pth'
    detector = generate_model('resnet')
    detector_data = torch.load(detector_path)
    detector.load_state_dict(detector_data['state_dict'])

    pretrain_path = 'model_weight/save_5.pth'
    model = generate_model('resnext_3bn')	
    model_data = torch.load(pretrain_path)
    model.load_state_dict(model_data['state_dict'])

    spatial_transform = Compose([Scale(opt.sample_size), CenterCrop(opt.sample_size), torchvision.transforms.ToTensor()])

    wrapped_model = MyPytorchClassifier(detector, model, spatial_transform)
	
    return wrapped_model

	