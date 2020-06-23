# This file basically runs Rectangular Occlusion Attacks (ROA) see paper 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import torchvision
from PIL import Image

#from utils import *
from spatial_transforms import Compose, Scale, CenterCrop


class ROA(object):

    def __init__(self, base_classifier, size):
        self.base_classifier = base_classifier
        self.img_size = size 
                
    # ---random_search--- #
    def random_search(self, X, y, num_frame, alpha, num_iter, width, height, xskip, yskip, random = False):

        size = self.img_size

        xtimes = (size-width) //xskip
        ytimes = (size-height)//yskip

        output_j = torch.randint(low=0, high=ytimes, size=(num_frame,))
        output_i = torch.randint(low=0, high=xtimes, size=(num_frame,))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd(X,y,width, height, alpha, num_iter, xskip, yskip, output_j, output_i, num_frame)
            

    def my_inside_pgd(self, X, y, width, height, alpha, num_iter, xskip, yskip, out_j, out_i, num_frame, random = False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):		
            j = out_j[k]
            i = out_i[k]
            sticker[0,:,k,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)+1/2            
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
        
        for t in range(num_iter):                          
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()
        return X1, X1*sticker


    # ---random_search_one--- #		
    def random_search_one(self, X, y, num_pixel, num_frame, alpha, num_iter):

        size = self.img_size

        pixel_pos = torch.randint(low=0, high=size, size=(num_frame, num_pixel, 2))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd_one(X, y, alpha, num_iter, pixel_pos, num_pixel, num_frame)
            

    def my_inside_pgd_one(self, X, y, alpha, num_iter, pixel_pos, num_pixel, num_frame, random=False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):
            for p in range(num_pixel):		
                yy = pixel_pos[k,p,0]
                xx = pixel_pos[k,p,1]
                sticker[0, :, k, yy, xx] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
	        
        for t in range(num_iter):                          
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()			
        return X1, X1*sticker
		
       
    # ---random_search_bn--- #
    def random_search_bn(self, X, type_roa, y, num_frame, alpha, num_iter, width, height, xskip, yskip, random = False):

        size = self.img_size

        xtimes = (size-width) //xskip
        ytimes = (size-height)//yskip

        output_j = torch.randint(low=0, high=ytimes, size=(num_frame,))
        output_i = torch.randint(low=0, high=xtimes, size=(num_frame,))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd_bn(X, type_roa, y, width, height, alpha, num_iter, xskip, yskip, output_j, output_i, num_frame)
            

    def my_inside_pgd_bn(self, X, type_roa, y, width, height, alpha, num_iter, xskip, yskip, out_j, out_i, num_frame, random = False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):		
            j = out_j[k]
            i = out_i[k]
            sticker[0,:,k,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)+1/2            
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
        
        for t in range(num_iter):                          
            loss = nn.CrossEntropyLoss()(model(X1, type_roa), y)  # ROA type=2
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()			
        return X1, X1*sticker

        
    # ---random_search_one_bn--- #		
    def random_search_one_bn(self, X, type_roa, y, num_pixel, num_frame, alpha, num_iter):

        size = self.img_size

        pixel_pos = torch.randint(low=0, high=size, size=(num_frame, num_pixel, 2))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd_one_bn(X, type_roa, y, alpha, num_iter, pixel_pos, num_pixel, num_frame)
            

    def my_inside_pgd_one_bn(self, X, type_roa, y, alpha, num_iter, pixel_pos, num_pixel, num_frame, random=False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):
            for p in range(num_pixel):		
                yy = pixel_pos[k,p,0]
                xx = pixel_pos[k,p,1]
                sticker[0, :, k, yy, xx] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
	        
        for t in range(num_iter):                          
            loss = nn.CrossEntropyLoss()(model(X1, type_roa), y)
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()			
        return X1, X1*sticker


    # ---random_search_comb--- #
    def random_search_comb(self, detector, X, y, num_frame, alpha, num_iter, width, height, xskip, yskip, random = False):

        size = self.img_size

        xtimes = (size-width) //xskip
        ytimes = (size-height)//yskip

        output_j = torch.randint(low=0, high=ytimes, size=(num_frame,))
        output_i = torch.randint(low=0, high=xtimes, size=(num_frame,))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd_comb(detector, X, y, width, height, alpha, num_iter, xskip, yskip, output_j, output_i, num_frame)
            

    def my_inside_pgd_comb(self, detector, X, y, width, height, alpha, num_iter, xskip, yskip, out_j, out_i, num_frame, random = False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):		
            j = out_j[k]
            i = out_i[k]
            sticker[0,:,k,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)+1/2            
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
        
        for t in range(num_iter):                          
            loss = nn.CrossEntropyLoss()(detector_and_model(detector, model, X1+0), y)  # ROA type=2
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()
        return X1, X1*sticker


    # ---random_search_one_comb--- #		
    def random_search_one_comb(self, detector, X, y, num_pixel, num_frame, alpha, num_iter):

        size = self.img_size

        pixel_pos = torch.randint(low=0, high=size, size=(num_frame, num_pixel, 2))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd_one_comb(detector, X, y, alpha, num_iter, pixel_pos, num_pixel, num_frame)
            

    def my_inside_pgd_one_comb(self, detector, X, y, alpha, num_iter, pixel_pos, num_pixel, num_frame, random=False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):
            for p in range(num_pixel):		
                yy = pixel_pos[k,p,0]
                xx = pixel_pos[k,p,1]
                sticker[0, :, k, yy, xx] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
	        
        for t in range(num_iter):                          
            loss = nn.CrossEntropyLoss()(detector_and_model(detector, model, X1+0), y)
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()			
        return X1, X1*sticker


    # ---random_search_ensemble--- #
    def random_search_ensemble(self, X, type_roa, y, num_frame, alpha, num_iter, width, height, xskip, yskip, random = False):

        size = self.img_size

        xtimes = (size-width) //xskip
        ytimes = (size-height)//yskip

        output_j = torch.randint(low=0, high=ytimes, size=(num_frame,))
        output_i = torch.randint(low=0, high=xtimes, size=(num_frame,))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd_ensemble(X, type_roa, y, width, height, alpha, num_iter, xskip, yskip, output_j, output_i, num_frame)
            

    def my_inside_pgd_ensemble(self, X, type_roa, y, width, height, alpha, num_iter, xskip, yskip, out_j, out_i, num_frame, random = False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):		
            j = out_j[k]
            i = out_i[k]
            sticker[0,:,k,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)+1/2            
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
        
        for t in range(num_iter):
            #print(t)		
            #print(torch.cat((X1, X1, X1), dim=0).shape)		
            loss = nn.CrossEntropyLoss()(model(torch.cat((X1, X1, X1), dim=0), type_roa), torch.cat((y, y, y), dim=0))
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()			
        return X1, X1*sticker

        
    # ---random_search_one_ensemble--- #		
    def random_search_one_ensemble(self, X, type_roa, y, num_pixel, num_frame, alpha, num_iter):

        size = self.img_size

        pixel_pos = torch.randint(low=0, high=size, size=(num_frame, num_pixel, 2))            
        
        with torch.set_grad_enabled(True):
            return self.my_inside_pgd_one_ensemble(X, type_roa, y, alpha, num_iter, pixel_pos, num_pixel, num_frame)
            

    def my_inside_pgd_one_ensemble(self, X, type_roa, y, alpha, num_iter, pixel_pos, num_pixel, num_frame, random=False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape)  # CANNOT requires_grad
        
        for k in range(num_frame):
            for p in range(num_pixel):		
                yy = pixel_pos[k,p,0]
                xx = pixel_pos[k,p,1]
                sticker[0, :, k, yy, xx] = 1            

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)
        else:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
	        
        for t in range(num_iter):	
            loss = nn.CrossEntropyLoss()(model(torch.cat((X1, X1, X1), dim=0), type_roa), torch.cat((y, y, y), dim=0))
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()			
        return X1, X1*sticker


def detector_and_model(detector, model, inputs):

    spatial_transform = Compose([Scale(112), CenterCrop(112), torchvision.transforms.ToTensor()])

    temp_root = '/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/image.jpg'
    frame_num = len(inputs[0,0,:,0,0])
    for kk in range(frame_num):	
        inputs_frame = torchvision.transforms.ToPILImage()(inputs[0,:,kk,:,:].cpu())
        inputs_frame.save(temp_root)
        inputs_frame = Image.open(temp_root).convert('RGB')		
        inputs_frame = spatial_transform(inputs_frame)
        inputs[0,:,kk,:,:] = inputs_frame
		
    outputs = detector(inputs)
    
    _, type_pred = outputs.topk(1, 1, True)
    type_pred = type_pred.t().cpu().numpy()
    
    outputs = model(inputs, type_pred[0,0])
    #print(type_pred[0,0])	
    
    return outputs
        
        