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

from advertorch.attacks.my_videoattack_ensemble import LinfPGDAttack_bn
from sticker_attack import ROA

	
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

    '''    
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
    '''		
		
    video_outputs_clean0 = []
    video_outputs_clean1 = []
    video_outputs_clean2 = []	
    video_outputs_pgd0 = []
    video_outputs_pgd1 = []
    video_outputs_pgd2 = []	
    video_outputs_roa0 = []
    video_outputs_roa1 = []
    video_outputs_roa2 = []	
    video_outputs_framing0 = []
    video_outputs_framing1 = []
    video_outputs_framing2 = []	
    video_outputs_one0 = []
    video_outputs_one1 = []
    video_outputs_one2 = []	
    for i, (inputs, segments) in enumerate(data_loader):

        inputs = Variable(inputs, volatile=True)		
		
        # roa
        adversary_roa = ROA(model, opt.sample_size)
        adv_inputs_roa, perturb_roa = adversary_roa.random_search_ensemble(inputs, 33, targets, opt.sparsity, opt.step_size, 
                                opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
        # one
        adversary_one = ROA(model, opt.sample_size)
        adv_inputs_one, perturb_one = adversary_one.random_search_one_ensemble(inputs, 33, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								
          
        # pgd
        adversary_pgd = LinfPGDAttack_bn(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
        adv_inputs_pgd, perturb_pgd = adversary_pgd.perturb(inputs, 33, sparse_map, targets)		

        # framing
        adversary_framing = LinfPGDAttack_bn(predict=model, loss_fn=criterion, 
		                       eps=float(255/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
        adv_inputs_framing, perturb_framing = adversary_framing.perturb(inputs, 33, framing_mask, targets)	
	
		
        outputs_clean0 = model(inputs, 0)
        outputs_clean1 = model(inputs, 1)
        outputs_clean2 = model(inputs, 2)			
        outputs_pgd0 = model(adv_inputs_pgd, 0)
        outputs_pgd1 = model(adv_inputs_pgd, 1)
        outputs_pgd2 = model(adv_inputs_pgd, 2)			
        outputs_roa0 = model(adv_inputs_roa, 0)
        outputs_roa1 = model(adv_inputs_roa, 1)
        outputs_roa2 = model(adv_inputs_roa, 2)			
        outputs_framing0 = model(adv_inputs_framing, 0)
        outputs_framing1 = model(adv_inputs_framing, 1)			
        outputs_framing2 = model(adv_inputs_framing, 2)
        outputs_one0 = model(adv_inputs_one, 0)
        outputs_one1 = model(adv_inputs_one, 1)
        outputs_one2 = model(adv_inputs_one, 2)		
		

        video_outputs_clean0.append(outputs_clean0.cpu().data)
        video_outputs_clean1.append(outputs_clean1.cpu().data)
        video_outputs_clean2.append(outputs_clean2.cpu().data)		
        video_outputs_pgd0.append(outputs_pgd0.cpu().data)
        video_outputs_pgd1.append(outputs_pgd1.cpu().data)
        video_outputs_pgd2.append(outputs_pgd2.cpu().data)				
        video_outputs_roa0.append(outputs_roa0.cpu().data)
        video_outputs_roa1.append(outputs_roa1.cpu().data)
        video_outputs_roa2.append(outputs_roa2.cpu().data)		
        video_outputs_framing0.append(outputs_framing0.cpu().data)
        video_outputs_framing1.append(outputs_framing1.cpu().data)
        video_outputs_framing2.append(outputs_framing2.cpu().data)				
        video_outputs_one0.append(outputs_one0.cpu().data)
        video_outputs_one1.append(outputs_one1.cpu().data)
        video_outputs_one2.append(outputs_one2.cpu().data)		
					
	
    video_outputs_clean0 = torch.cat(video_outputs_clean0)
    video_outputs_clean1 = torch.cat(video_outputs_clean1)
    video_outputs_clean2 = torch.cat(video_outputs_clean2)	
    video_outputs_pgd0 = torch.cat(video_outputs_pgd0)
    video_outputs_pgd1 = torch.cat(video_outputs_pgd1)
    video_outputs_pgd2 = torch.cat(video_outputs_pgd2)
    video_outputs_roa0 = torch.cat(video_outputs_roa0)
    video_outputs_roa1 = torch.cat(video_outputs_roa1)
    video_outputs_roa2 = torch.cat(video_outputs_roa2)
    video_outputs_framing0 = torch.cat(video_outputs_framing0)
    video_outputs_framing1 = torch.cat(video_outputs_framing1)
    video_outputs_framing2 = torch.cat(video_outputs_framing2)	
    video_outputs_one0 = torch.cat(video_outputs_one0)
    video_outputs_one1 = torch.cat(video_outputs_one1)
    video_outputs_one2 = torch.cat(video_outputs_one2)	

    _, max_indices_clean0 = video_outputs_clean0.max(dim=1)
    _, max_indices_clean1 = video_outputs_clean1.max(dim=1)
    _, max_indices_clean2 = video_outputs_clean2.max(dim=1)		
    _, max_indices_pgd0 = video_outputs_pgd0.max(dim=1)
    _, max_indices_pgd1 = video_outputs_pgd1.max(dim=1)
    _, max_indices_pgd2 = video_outputs_pgd2.max(dim=1)	
    _, max_indices_roa0 = video_outputs_roa0.max(dim=1)
    _, max_indices_roa1 = video_outputs_roa1.max(dim=1)
    _, max_indices_roa2 = video_outputs_roa2.max(dim=1)		
    _, max_indices_framing0 = video_outputs_framing0.max(dim=1)
    _, max_indices_framing1 = video_outputs_framing1.max(dim=1)
    _, max_indices_framing2 = video_outputs_framing2.max(dim=1)	
    _, max_indices_one0 = video_outputs_one0.max(dim=1)
    _, max_indices_one1 = video_outputs_one1.max(dim=1)
    _, max_indices_one2 = video_outputs_one2.max(dim=1)	
    

    return class_names[max_indices_clean0[i]], class_names[max_indices_pgd0[i]], class_names[max_indices_roa0[i]], class_names[max_indices_framing0[i]], class_names[max_indices_one0[i]], class_names[max_indices_clean1[i]], class_names[max_indices_pgd1[i]], class_names[max_indices_roa1[i]], class_names[max_indices_framing1[i]], class_names[max_indices_one1[i]], class_names[max_indices_clean2[i]], class_names[max_indices_pgd2[i]], class_names[max_indices_roa2[i]], class_names[max_indices_framing2[i]], class_names[max_indices_one2[i]]		   

		