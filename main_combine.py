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

from spatial_transforms import Compose, Scale, CenterCrop
from temporal_transforms import LoopPadding
from dataset import Video, Video_detect, Video_detect_open
from opts_detect import parse_opts
from model import generate_model
from mean import get_mean
from utils import *
from openmax import *

from advertorch.attacks.my_videoattack_comb import LinfPGDAttack_comb
from sticker_attack import ROA

if __name__=="__main__":
    opt = parse_opts()
    opt.n_classes = 3  # for test
	
    detector = generate_model(opt)
    print('loading detector {}'.format(opt.detector_path))
    detector_data = torch.load(opt.detector_path)
    detector.load_state_dict(detector_data['state_dict'])
    detector.eval()

    opt.model_name = 'resnext_3bn'
    opt.model_depth	= 101
    opt.resnet_shortcut = 'B'
    opt.n_classes = 101
    model = generate_model(opt)	
    print('loading model {}'.format(opt.model_name))
    model_data = torch.load(opt.pretrain_path)
    model.load_state_dict(model_data['state_dict'])
    model.eval()    

    #train_files = []
    #train_dir = '/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/ucfTrainTestlist/testlist02_jpg.txt'	
    #with open(train_dir, 'r') as f:
    #    for row in f:
    #        train_files.append(row[:-1])
			
    test_files = []
    #test_dir = '/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/ucfTrainTestlist/testlist03b_jpg.txt'
    test_dir = opt.input
    with open(test_dir, 'r') as f:
        for row in f:
            test_files.append(row[:-1])			

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

	# === load data  === #			
    spatial_transform = Compose([Scale(opt.sample_size), CenterCrop(opt.sample_size), torchvision.transforms.ToTensor()])
    temporal_transform = LoopPadding(opt.sample_duration)
											  			
	# === save information in folder === #
    save_dir = '/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/results_comb'
    saveDoc = save_dir + '/' + opt.exp_name + '_model.txt'
    saveDoc2 = save_dir + '/' + opt.exp_name + '_detector.txt'    
    print('=> will save everything to {}'.format(saveDoc))	 
    with open(saveDoc, "a") as myfile:
        myfile.writelines(str(opt) + '\n\n')	
        LL = ['-test data: '+str(opt.input)+'\n','-model: '+str(opt.pretrain_path)+'\n','-detector: '+str(opt.detector_path)+'\n\n']	
        myfile.writelines(LL)
        time_start = datetime.datetime.now()
        myfile.write(str(time_start) + '\n\n')	    

    '''
	# === open-set === # 
    open_true = np.ones(535*5)
    open_true[0:535*3] = 0  
    openmax_scores = openset_weibull(test_loader, train_loader, model)
    openmax_scores2 = openset_softmax_confidence(test_loader, model)
    auc_score = plot_roc(open_true, openmax_scores)  # 0.611
    auc_score2 = plot_roc(open_true, openmax_scores2)  # 0.607
    print(auc_score)
    print(auc_score2)
    '''
	
	# === run epoch === #
    #'''
    if opt.attack_type=='clean':
        target_detect = 0
    if opt.attack_type=='noise' or opt.attack_type=='one':
        target_detect = 1        
    if opt.attack_type=='roa' or opt.attack_type=='framing':
        target_detect = 2
        
    count = 0
    correct = 0
    correct_detect = 0    
    for test_file in test_files:
        count = count + 1
    
        video_path = os.path.join(opt.video_root, test_file)
        test_data = Video(video_path, spatial_transform=spatial_transform, temporal_transform=temporal_transform, sample_duration=opt.sample_duration) 
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.n_threads, pin_memory=True)
 
        for i, (inputs, segments) in enumerate(test_loader):
        
            label = video_path.split('/v_')[0].split('_jpg/')[1]
            for i in range(len(class_names)):
                class_i = class_names[i]
                if label == class_i.split(' ')[1]:
                   break           
            targets = torch.zeros(1).long().cuda()
            targets[0] = i        
            inputs = Variable(inputs, volatile=True)

            if opt.attack_type == 'clean':
                class_pred, type_pred = detector_and_model(detector, model, inputs, spatial_transform)
            else:                
                #adv_inputs, perturb = attacker_comb(inputs, targets, detector, model, opt)
                adv_inputs, perturb = attacker_bn(inputs, targets, model, opt)                
                class_pred, type_pred = detector_and_model(detector, model, adv_inputs, spatial_transform)

        if opt.save_image:		
            visual_results(adv_inputs, perturb, opt.sample_duration, video_path, opt)	                

        # model evaluation
        if class_pred == targets:
            correct = correct + 1
        acc = correct / count * 100        
        with open(saveDoc, "a") as myfile:
            myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc) + class_names[targets] + ', predict: ' + class_names[class_pred] + '\n')				
        print('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc) + class_names[targets]	 + ', predict: ' + class_names[class_pred])
        
        # detector evaluation        
        if type_pred == target_detect:
            correct_detect = correct_detect + 1
        acc_detect = correct_detect / count * 100       
        target_name = idx_to_name(target_detect)
        pred_name = idx_to_name(type_pred)            
        with open(saveDoc2, "a") as myfile:	
             myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_detect) + target_name + ', predict: ' + pred_name + '\n')
        #print('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_detect) + target_name + ', predict: ' + pred_name)			 
    
    with open(saveDoc, "a") as myfile:
        myfile.write('\n' + str(datetime.datetime.now()-time_start) + ',       ' + str(datetime.datetime.now()) + '\n')    
	#'''

    