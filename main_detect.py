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
from model_detect import generate_model_detect
from mean import get_mean
from utils import Logger, AverageMeter, calculate_accuracy

if __name__=="__main__":
    opt = parse_opts()
	
    model, parameters = generate_model_detect(opt)	
    print('loading model {}'.format(opt.model_name))

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
    save_dir = '/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/video_classification/detect_train'	
    save_dir = os.path.join(save_dir, opt.exp_name)
    if not os.path.exists(save_dir):
       os.mkdir(save_dir)
    saveDoc = save_dir + '/opt.txt' 	 
    with open(saveDoc, "w") as myfile:	
        myfile.writelines(str(opt) + '\n')
    train_logger = Logger(os.path.join(save_dir, 'train.log'),['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(os.path.join(save_dir, 'train_batch.log'), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    val_logger = Logger(os.path.join(save_dir, 'val.log'), ['epoch', 'loss', 'acc'])		

	# === optimizer === # 	
    optimizer = optim.SGD(parameters, lr=opt.learning_rate, momentum=opt.momentum, dampening=opt.dampening,
        weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
	
    criterion = nn.CrossEntropyLoss().cuda()	
	
	# === run epoch === #   
    for epoch in range(opt.n_epochs):
        epoch = epoch + 1
	
        # Training
        print('train at epoch {}'.format(epoch))        
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()        
        end_time = time.time()		
        for i, (inputs, targets) in enumerate(train_loader):
		    
            #if i>=3:
            #    break			
	    		 
            data_time.update(time.time() - end_time)
            
            targets = targets.cuda(async=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
			
            outputs = model(inputs)		
            loss = criterion(outputs, targets)			
            acc = calculate_accuracy(outputs, targets)		

            losses.update(loss, inputs.size(0))		
            accuracies.update(acc, inputs.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            train_batch_logger.log({'epoch': epoch, 'batch': i + 1, 'iter': (epoch - 1) * len(train_loader) + (i + 1),
                'loss': losses.val, 'acc': accuracies.val, 'lr': optimizer.param_groups[0]['lr']})
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=accuracies))

        train_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg, 'lr': optimizer.param_groups[0]['lr']})
        
        if epoch % opt.checkpoint == 0:
            save_file_path = os.path.join(save_dir, 'save_{}.pth'.format(epoch))
            states = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(states, save_file_path)			

        # Testing			
        print('validation at epoch {}'.format(epoch))
        
        model.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()      
        end_time = time.time()
        for i, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end_time)
        
            targets = targets.cuda(async=True)
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
	    	
            outputs = model(inputs)	    		
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
        
            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))
        
            batch_time.update(time.time() - end_time)
            end_time = time.time()
        
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i + 1, len(test_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=accuracies))
        
        val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        scheduler.step(losses.avg)		
	
	