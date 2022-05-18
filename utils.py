import csv
import os
import sys
import json
import subprocess
import datetime
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision
from PIL import Image

from spatial_transforms import Compose, Scale, CenterCrop
from dataset import Video_detect, Video_detect_open
from opts_kk import parse_opts
from model import generate_model
from model_detect import generate_model_detect
from mean import get_mean, get_mean_std
from utils import *

from advertorch.attacks.my_videoattack_bn import LinfPGDAttack_bn
from advertorch.attacks.my_videoattack_comb import LinfPGDAttack_comb
from sticker_attack import ROA


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum()#.data[0]

    return n_correct_elems / batch_size


def calculate_accuracy_mine(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum()#.data[0]

    return n_correct_elems / batch_size, pred


def get_opt(args=None):
    opt = parse_opts(args)

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    return opt


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model    

    
def idx_to_name(idx):
    if idx==0:
        name = 'clean'    
    elif idx==1:
        name = 'PGD'
    elif idx==2:
        name = 'ROA'
        
    return name
    
 
def detector_and_model(detector, model, inputs, spatial_transform):

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

    _, class_pred = outputs.topk(1, 1, True)
    class_pred = class_pred.t().cpu().numpy()
    
    return class_pred[0,0], type_pred[0,0]
    

def attacker_comb(inputs, targets, detector, model, opt):

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # attack mask	
    sparse_map = torch.ones((1,3,opt.sample_duration,opt.sample_size,opt.sample_size))
    sparse_map[:,:,opt.sparsity:,:,:] = 0
    framing_mask = torch.zeros((1,3,opt.sample_duration,opt.sample_size-opt.framing_width*2,opt.sample_size-opt.framing_width*2))
    p2d = (opt.framing_width, opt.framing_width, opt.framing_width, opt.framing_width)
    framing_mask = F.pad(framing_mask, p2d, 'constant', 1)
    framing_mask[:,:,opt.sparsity:,:,:] = 0			
    if opt.attack_type == 'noise':
        attack_mask = sparse_map
        opt.epsilon = 4		
    else:
        attack_mask = framing_mask	
        opt.epsilon = 255

    if opt.attack_type == 'roa':
        adversary = ROA(model, opt.sample_size)
        adv_inputs, perturb = adversary.random_search_comb(detector, inputs, targets, opt.sparsity, opt.step_size, 
                            opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
    elif opt.attack_type == 'one':
        adversary = ROA(model, opt.sample_size)
        adv_inputs, perturb = adversary.random_search_one_comb(detector, inputs, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								                
    else:
        adversary = LinfPGDAttack_comb(predict=model, loss_fn=criterion, 
		                   eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
        adv_inputs, perturb = adversary.perturb(detector, inputs, attack_mask, targets)		
		    
    return adv_inputs, perturb


def attacker_bn(inputs, targets, model, opt):

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # attack mask	
    sparse_map = torch.ones((1,3,opt.sample_duration,opt.sample_size,opt.sample_size))
    sparse_map[:,:,opt.sparsity:,:,:] = 0
    framing_mask = torch.zeros((1,3,opt.sample_duration,opt.sample_size-opt.framing_width*2,opt.sample_size-opt.framing_width*2))
    p2d = (opt.framing_width, opt.framing_width, opt.framing_width, opt.framing_width)
    framing_mask = F.pad(framing_mask, p2d, 'constant', 1)
    framing_mask[:,:,opt.sparsity:,:,:] = 0			
    if opt.attack_type == 'noise':
        attack_mask = sparse_map
        opt.epsilon = 4		
    else:
        attack_mask = framing_mask	
        opt.epsilon = 255

    # attack type number
    if opt.attack_type=='clean':
        attack_bn = 0
    elif opt.attack_type=='noise' or opt.attack_type=='one':		
        attack_bn = 1
    elif opt.attack_type=='roa' or opt.attack_type=='framing':
        attack_bn = 2

    if opt.attack_type == 'roa':
        adversary = ROA(model, opt.sample_size)
        adv_inputs, perturb = adversary.random_search_bn(inputs, attack_bn, targets, opt.sparsity, opt.step_size, 
                                    opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
    elif opt.attack_type == 'one':
        adversary = ROA(model, opt.sample_size)
        adv_inputs, perturb = adversary.random_search_one_bn(inputs, attack_bn, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								                
    else:
        adversary = LinfPGDAttack_bn(predict=model, loss_fn=criterion, 
		                   eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
        adv_inputs, perturb = adversary.perturb(inputs, attack_bn, attack_mask, targets)		
		    
    return adv_inputs, perturb
    

def visual_results(adv_frames, perturb, frame_num, video_dir, opt):

    this_dir = '/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification'
    save_dir = this_dir + '/visual_results/v_' + video_dir.split('/v_')[1]
    if not os.path.exists(save_dir):
       os.mkdir(save_dir)
       os.mkdir(save_dir + '/adv_frame')
       os.mkdir(save_dir + '/perturb')	   

    if opt.attack_type == 'noise':	   
        perturb = perturb * 20
	
    for kk in range(frame_num):
	
        adv_frames_img = torchvision.transforms.ToPILImage()(adv_frames[0,:,kk,:,:].cpu())
        perturb_img = torchvision.transforms.ToPILImage()(perturb[0,:,kk,:,:].cpu())	
       
        adv_frames_img.save(save_dir + '/adv_frame/frame_' + f'{kk:02}' + '.jpg')
        perturb_img.save(save_dir + '/perturb/frame_' + f'{kk:02}' + '.jpg')

	
def visual_results_2(adv_frames, perturb, frame_num, video_dir, opt):

    this_dir = '/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification'
    save_dir = this_dir + '/visual_results/for_paper/One_comb' + video_dir.split('/v_')[0].split('_jpg')[1]
    save_dir2 = save_dir + '/v_' + video_dir.split('/v_')[1]
    if not os.path.exists(save_dir):
       os.mkdir(save_dir)
    if not os.path.exists(save_dir2):
       os.mkdir(save_dir2)	   
	
    for kk in range(frame_num):	
        adv_frames_img = torchvision.transforms.ToPILImage()(adv_frames[0,:,kk,:,:].cpu())
        jj = kk + 1		
        adv_frames_img.save(save_dir2 + '/image_' + f'{jj:05}' + '.jpg')

        