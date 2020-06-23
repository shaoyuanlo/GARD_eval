import os
import numpy as np
from PIL import Image 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding

from advertorch.attacks.my_videoattack import LinfPGDAttack
from advertorch.attacks.my_videoattack_bn import LinfPGDAttack_bn
from advertorch.attacks.my_videoattack_mul import LinfPGDAttack_mul
from sticker_attack import ROA

def classify_video(video_dir, video_name, class_names, model, opt):
    assert opt.mode in ['score', 'feature']

    #spatial_transform = Compose([Scale(opt.sample_size), CenterCrop(opt.sample_size), ToTensor(), Normalize(opt.mean, [1, 1, 1])])
    spatial_transform = Compose([Scale(opt.sample_size), CenterCrop(opt.sample_size), torchvision.transforms.ToTensor()])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)

    video_outputs = []
    video_segments = []
    for i, (inputs, segments) in enumerate(data_loader):	
	
        inputs = Variable(inputs, volatile=True)

        if opt.model_name=='resnext_3bn':		
            outputs = model(inputs, 0)
        else:		
            outputs = model(inputs)			

        video_outputs.append(outputs.cpu().data)
        video_segments.append(segments)

    if opt.save_image:		
        visual_results_2(inputs, inputs, opt.sample_duration, video_dir, opt)		
    video_outputs = torch.cat(video_outputs)
    video_segments = torch.cat(video_segments)
    results = {'video': video_name,'clips': []}

    _, max_indices = video_outputs.max(dim=1)
    for i in range(video_outputs.size(0)):
        #clip_results = {'segment': video_segments[i].tolist(),}
        clip_results = {'label': class_names[max_indices[i]]}		

        if opt.mode == 'score':
            #clip_results['label'] = class_names[max_indices[i]]
            clip_results['scores'] = video_outputs[i].max()
        elif opt.mode == 'feature':
            clip_results['features'] = video_outputs[i].tolist()

        results['clips'].append(clip_results)

    #return results
    return class_names[max_indices[i]]	

	
def classify_video_adv(video_dir, video_name, class_names, model, label, opt):
    assert opt.mode in ['score', 'feature']

    #spatial_transform = Compose([Scale(opt.sample_size), CenterCrop(opt.sample_size), ToTensor(), Normalize(opt.mean, [1, 1, 1])])
    spatial_transform = Compose([Scale(opt.sample_size), CenterCrop(opt.sample_size), torchvision.transforms.ToTensor()])	
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    for i in range(len(class_names)):
        class_i = class_names[i]
        if label == class_i.split(' ')[1]:
            break           
    targets = torch.zeros(1).long().cuda()
    targets[0] = i

    # attack mask	
    sparse_map = torch.ones((1,3,opt.sample_duration,opt.sample_size,opt.sample_size))
    sparse_map[:,:,opt.sparsity:,:,:] = 0
    framing_mask = torch.zeros((1,3,opt.sample_duration,opt.sample_size-opt.framing_width*2,opt.sample_size-opt.framing_width*2))
    p2d = (opt.framing_width, opt.framing_width, opt.framing_width, opt.framing_width)
    framing_mask = F.pad(framing_mask, p2d, 'constant', 1)
    framing_mask[:,:,opt.sparsity:,:,:] = 0			
    if opt.attack_type == 'noise':
        attack_mask = sparse_map		
    else:
        attack_mask = framing_mask	
        opt.epsilon = 255	

    # attack type number
    if opt.attack_bn=='clean':
        attack_bn = 0
    elif opt.attack_bn=='noise':		
        attack_bn = 1
    elif opt.attack_bn=='roa':
        attack_bn = 2
    if opt.inf_bn=='clean':
        inf_bn = 0
    elif opt.inf_bn=='noise':		
        inf_bn = 1
    elif opt.inf_bn=='roa':
        inf_bn = 2		
		
    video_outputs = []	
    for i, (inputs, segments) in enumerate(data_loader):

        inputs = Variable(inputs, volatile=True)

        if opt.model_name=='resnext_3bn':		
            if opt.attack_type == 'roa':
                adversary = ROA(model, opt.sample_size)
                adv_inputs, perturb = adversary.random_search_bn(inputs, attack_bn, targets, opt.sparsity, opt.step_size, 
                                    opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
            elif opt.attack_type == 'one':
                adversary = ROA(model, opt.sample_size)
                adv_inputs, perturb = adversary.random_search_one_bn(inputs, attack_bn, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								
            elif opt.attack_type == 'mul':
                adversary = LinfPGDAttack_mul(predict=model, loss_fn=criterion, 
		                           eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
                adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)                
            else:
                adversary = LinfPGDAttack_bn(predict=model, loss_fn=criterion, 
		                           eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
                adv_inputs, perturb = adversary.perturb(inputs, attack_bn, attack_mask, targets)		
		    
            outputs = model(adv_inputs, inf_bn)
			
        else:		
            if opt.attack_type == 'roa':
                adversary = ROA(model, opt.sample_size)
                adv_inputs, perturb = adversary.random_search(inputs, targets, opt.sparsity, opt.step_size, 
                                    opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
            elif opt.attack_type == 'one':
                adversary = ROA(model, opt.sample_size)
                adv_inputs, perturb = adversary.random_search_one(inputs, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								
            elif opt.attack_type == 'mul':
                adversary = LinfPGDAttack_mul(predict=model, loss_fn=criterion, 
		                           eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
                adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)                
            else:
                adversary = LinfPGDAttack(predict=model, loss_fn=criterion, 
		                           eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
                adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)		
		    
            outputs = model(adv_inputs)			

        video_outputs.append(outputs.cpu().data)

    if opt.save_image:		
        visual_results(adv_inputs, perturb, opt.sample_duration, video_dir, opt)		
    video_outputs = torch.cat(video_outputs)
    results = {'video': video_name,'clips': []}

    _, max_indices = video_outputs.max(dim=1)
	
    for i in range(video_outputs.size(0)):

        clip_results = {'label': class_names[max_indices[i]]}		

        if opt.mode == 'score':
            clip_results['scores'] = video_outputs[i].max()
        elif opt.mode == 'feature':
            clip_results['features'] = video_outputs[i].tolist()

        results['clips'].append(clip_results)

    return class_names[max_indices[i]]


def visual_results(adv_frames, perturb, frame_num, video_dir, opt):

    this_dir = '/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/video_classification'
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

    this_dir = '/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/video_classification'
    save_dir = this_dir + '/visual_results/Clean_test' + video_dir.split('/v_')[0].split('_jpg')[1]
    save_dir2 = save_dir + '/v_' + video_dir.split('/v_')[1]
    if not os.path.exists(save_dir):
       os.mkdir(save_dir)
    if not os.path.exists(save_dir2):
       os.mkdir(save_dir2)	   
	
    for kk in range(frame_num):	
        adv_frames_img = torchvision.transforms.ToPILImage()(adv_frames[0,:,kk,:,:].cpu())
        jj = kk + 1		
        adv_frames_img.save(save_dir2 + '/image_' + f'{jj:05}' + '.jpg')

		