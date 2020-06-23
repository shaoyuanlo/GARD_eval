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
		
    video_outputs_clean = []
    video_outputs_pgd = []
    video_outputs_roa = []
    video_outputs_framing = []
    video_outputs_one = []	
    for i, (inputs, segments) in enumerate(data_loader):

        inputs = Variable(inputs, volatile=True)

        if opt.model_name=='resnext_3bn' or opt.model_name=='wideresnet_3bn':		
            # roa
            adversary_roa = ROA(model, opt.sample_size)
            adv_inputs_roa, perturb_roa = adversary_roa.random_search_bn(inputs, 2, targets, opt.sparsity, opt.step_size, 
                                    opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
            # one
            adversary_one = ROA(model, opt.sample_size)
            adv_inputs_one, perturb_one = adversary_one.random_search_one_bn(inputs, 1, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								
              
            # pgd
            adversary_pgd = LinfPGDAttack_bn(predict=model, loss_fn=criterion, 
		                           eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
            adv_inputs_pgd, perturb_pgd = adversary_pgd.perturb(inputs, 1, sparse_map, targets)		

            # framing
            adversary_framing = LinfPGDAttack_bn(predict=model, loss_fn=criterion, 
		                           eps=float(255/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
            adv_inputs_framing, perturb_framing = adversary_framing.perturb(inputs, 2, framing_mask, targets)	

		    
            outputs_clean = model(inputs, 0)
            outputs_pgd = model(adv_inputs_pgd, 1)
            outputs_roa = model(adv_inputs_roa, 2)
            outputs_framing = model(adv_inputs_framing, 2)
            outputs_one = model(adv_inputs_one, 1)
			
        else:		
            # roa
            adversary_roa = ROA(model, opt.sample_size)
            adv_inputs_roa, perturb_roa = adversary_roa.random_search(inputs, targets, opt.sparsity, opt.step_size, 
                                    opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
            # one
            adversary_one = ROA(model, opt.sample_size)
            adv_inputs_one, perturb_one = adversary_one.random_search_one(inputs, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								
            
            # pgd
            adversary_pgd = LinfPGDAttack(predict=model, loss_fn=criterion, 
		                           eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
            adv_inputs_pgd, perturb_pgd = adversary_pgd.perturb(inputs, sparse_map, targets)		

            # framing
            adversary_framing = LinfPGDAttack(predict=model, loss_fn=criterion, 
		                           eps=float(255/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
            adv_inputs_framing, perturb_framing = adversary_framing.perturb(inputs, framing_mask, targets)	

		    
            outputs_clean = model(inputs)
            outputs_pgd = model(adv_inputs_pgd)
            outputs_roa = model(adv_inputs_roa)
            outputs_framing = model(adv_inputs_framing)
            outputs_one = model(adv_inputs_one)		

        video_outputs_clean.append(outputs_clean.cpu().data)
        video_outputs_pgd.append(outputs_pgd.cpu().data)
        video_outputs_roa.append(outputs_roa.cpu().data)
        video_outputs_framing.append(outputs_framing.cpu().data)
        video_outputs_one.append(outputs_one.cpu().data)
		
        if opt.save_image:		
            visual_results_2(inputs, perturb_pgd, opt.sample_duration, video_dir, opt, 'Clean')
            visual_results_2(adv_inputs_pgd, perturb_pgd, opt.sample_duration, video_dir, opt, 'PGD')
            visual_results_2(adv_inputs_roa, perturb_pgd, opt.sample_duration, video_dir, opt, 'ROA')
            visual_results_2(adv_inputs_framing, perturb_pgd, opt.sample_duration, video_dir, opt, 'Framing')
            visual_results_2(adv_inputs_one, perturb_pgd, opt.sample_duration, video_dir, opt, 'One')			
	
    video_outputs_clean = torch.cat(video_outputs_clean)
    video_outputs_pgd = torch.cat(video_outputs_pgd)
    video_outputs_roa = torch.cat(video_outputs_roa)
    video_outputs_framing = torch.cat(video_outputs_framing)
    video_outputs_one = torch.cat(video_outputs_one)

    _, max_indices_clean = video_outputs_clean.max(dim=1)
    _, max_indices_pgd = video_outputs_pgd.max(dim=1)
    _, max_indices_roa = video_outputs_roa.max(dim=1)
    _, max_indices_framing = video_outputs_framing.max(dim=1)
    _, max_indices_one = video_outputs_one.max(dim=1)
    

    return class_names[max_indices_clean[i]], class_names[max_indices_pgd[i]], class_names[max_indices_roa[i]], class_names[max_indices_framing[i]], class_names[max_indices_one[i]]
    #return class_names[max_indices_clean[i]], class_names[max_indices_pgd[i]], class_names[max_indices_roa[i]]


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


def visual_results_2(adv_frames, perturb, frame_num, video_dir, opt, namee):

    this_dir = '/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/video_classification'
    save_dir = this_dir + '/visual_results/detector_data_wide/' + namee + '_train' + video_dir.split('/v_')[0].split('_jpg')[1]
    save_dir2 = save_dir + '/v_' + video_dir.split('/v_')[1]
    if not os.path.exists(save_dir):
       os.mkdir(save_dir)
    if not os.path.exists(save_dir2):
       os.mkdir(save_dir2)	   
	
    for kk in range(frame_num):	
        adv_frames_img = torchvision.transforms.ToPILImage()(adv_frames[0,:,kk,:,:].cpu())
        jj = kk + 1		
        adv_frames_img.save(save_dir2 + '/image_' + f'{jj:05}' + '.jpg')

		