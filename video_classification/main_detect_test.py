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
from dataset import Video_detect, Video_detect_open
from opts_detect import parse_opts
from model import generate_model
from mean import get_mean
from utils import *
from openmax import *

if __name__=="__main__":
    opt = parse_opts()
    opt.n_classes = 3  # for test
	
    model = generate_model(opt)
    print('loading model {}'.format(opt.detector_path))
    model_data = torch.load(opt.detector_path)
    model.load_state_dict(model_data['state_dict'])
    model.eval()    

    train_files = []
    train_dir = '/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/video_classification/ucfTrainTestlist/testlist02_jpg.txt'	
    with open(train_dir, 'r') as f:
        for row in f:
            train_files.append(row[:-1])
			
    test_files = []
    test_dir = '/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/video_classification/ucfTrainTestlist/testlist01_test_jpg.txt'	
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
	
    training_data = Video_detect(train_files, 'train', spatial_transform=spatial_transform, temporal_transform=temporal_transform, sample_duration=opt.sample_duration)
    test_data = Video_detect_open(test_files, spatial_transform=spatial_transform, temporal_transform=temporal_transform, sample_duration=opt.sample_duration)
	
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_threads, pin_memory=True)
											  			
	# === save information in folder === #
    save_dir = '/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/video_classification/detect_test'
    saveDoc = save_dir + '/' +str(opt.exp_name) + '.txt' 	 
    with open(saveDoc, "a") as myfile:	
        myfile.writelines(str(opt) + '\n\n')        

    #'''
	# === open-set === # 
    open_true = np.ones(1063*5)
    open_true[0:1063*3] = 0  
    openmax_scores = openset_weibull(test_loader, train_loader, model)
    openmax_scores2 = openset_softmax_confidence(test_loader, model)
    auc_score = plot_roc(open_true, openmax_scores)  # 0.7751
    auc_score2 = plot_roc(open_true, openmax_scores2)  # 0.7740
    print(auc_score)
    print(auc_score2)
    #'''
	
	# === run epoch === # 
    criterion = nn.CrossEntropyLoss().cuda()
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()      
    end_time = time.time()
    '''    
    for i, (inputs, targets) in enumerate(test_loader):
        data_time.update(time.time() - end_time)
        
        targets = targets.cuda(async=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
	    	
        outputs = model(inputs)        
        #print(outputs)
        #jjj=kkk
        
        loss = criterion(outputs, targets)
        acc, pred = calculate_accuracy_mine(outputs, targets)
        
        losses.update(loss.data, inputs.size(0))
        accuracies.update(acc, inputs.size(0))
  
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        for j in range(opt.batch_size):
            target_name = idx_to_name(targets[j].cpu().numpy())
            pred_name = idx_to_name(pred[0,j].cpu().numpy())
            
            with open(saveDoc, "a") as myfile:	
                myfile.writelines('id: ' + f'{i*opt.batch_size+j+1:02}' + ', acc: {0:.2f}, label: '.format(accuracies.avg*100) + target_name + ', predict: ' + pred_name + '\n')       
                print('id: ' + f'{i*opt.batch_size+j+1:02}' + ', acc: {0:.2f}, label: '.format(accuracies.avg*100) + target_name + ', predict: ' + pred_name)
	'''
    