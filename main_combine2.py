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
#from openmax import *

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
    opt.model_depth = 101
    #opt.model_name = 'wideresnet_3bn'
    #opt.model_depth = 50	
    opt.resnet_shortcut = 'B'
    opt.n_classes = 101
    model = generate_model(opt)	
    print('loading model {}'.format(opt.model_name))
    model_data = torch.load(opt.pretrain_path)
    model.load_state_dict(model_data['state_dict'])
    model.eval()    

    '''	
    train_files = []
    train_dir = '/home/sylo/SegNet/3D-ResNets-PyTorch/video_classification/ucfTrainTestlist/testlist02_jpg.txt'	
    with open(train_dir, 'r') as f:
        for row in f:
            train_files.append(row[:-1])
    '''
			
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
    saveDoc = save_dir + '/' + opt.exp_name + '_union.txt'
    saveDoc_clean = save_dir + '/' + opt.exp_name + '_clean.txt' 
    saveDoc_pgd = save_dir + '/' + opt.exp_name + '_pgd.txt' 
    saveDoc_roa = save_dir + '/' + opt.exp_name + '_roa.txt' 
    saveDoc_framing = save_dir + '/' + opt.exp_name + '_framing.txt' 
    saveDoc_one = save_dir + '/' + opt.exp_name + '_one.txt'	
    #saveDoc2 = save_dir + '/' + opt.exp_name + '_detector.txt'    
    print('=> will save everything to {}'.format(saveDoc))	 
    with open(saveDoc, "a") as myfile:
        myfile.writelines(str(opt) + '\n\n')	
        LL = ['-test data: '+str(opt.input)+'\n','-model: '+str(opt.pretrain_path)+'\n','-detector: '+str(opt.detector_path)+'\n\n']	
        myfile.writelines(LL)
        time_start = datetime.datetime.now()
        myfile.write(str(time_start) + '\n\n')
    with open(saveDoc_clean, "a") as myfile:
        myfile.writelines('Clean\n\n')
    with open(saveDoc_pgd, "a") as myfile:
        myfile.writelines('PGD\n\n')
    with open(saveDoc_roa, "a") as myfile:
        myfile.writelines('ROA\n\n')
    with open(saveDoc_framing, "a") as myfile:
        myfile.writelines('Framing\n\n')
    with open(saveDoc_one, "a") as myfile:
        myfile.writelines('One\n\n')		    

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
    '''
    if opt.attack_type=='clean':
        target_detect = 0
    if opt.attack_type=='noise' or opt.attack_type=='one':
        target_detect = 1        
    if opt.attack_type=='roa' or opt.attack_type=='framing':
        target_detect = 2
    '''
    id = 0        
    count = 0
    correct_clean = 0
    correct_pgd = 0
    correct_roa = 0
    correct_framing = 0
    correct_one = 0
    correct_union = 0
    type_correct_clean = 0
    type_correct_pgd = 0
    type_correct_roa = 0
    type_correct_framing = 0
    type_correct_one = 0    
    for test_file in test_files:
        id = id + 1
        if id <= 1523:
            print(id)
            continue			
	
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

            opt.attack_type = 'noise'
            #print('noise')			
            #adv_inputs_pgd, perturb_pgd = attacker_bn(inputs, targets, model, opt)
            adv_inputs_pgd, perturb_pgd = attacker_comb(inputs, targets, detector, model, opt)			
            opt.attack_type = 'roa'
            #print('roa')			
            #adv_inputs_roa, perturb_roa = attacker_bn(inputs, targets, model, opt)
            adv_inputs_roa, perturb_roa = attacker_comb(inputs, targets, detector, model, opt)			
            opt.attack_type = 'framing'
            #print('framing')			
            #adv_inputs_fra, perturb_fra = attacker_bn(inputs, targets, model, opt)
            adv_inputs_fra, perturb_fra = attacker_comb(inputs, targets, detector, model, opt)			
            opt.attack_type = 'one'
            #print('one')			
            #adv_inputs_one, perturb_one = attacker_bn(inputs, targets, model, opt)
            adv_inputs_one, perturb_one = attacker_comb(inputs, targets, detector, model, opt)			
			
            class_pred_clean, type_pred_clean = detector_and_model(detector, model, inputs, spatial_transform)		
            class_pred_pgd, type_pred_pgd = detector_and_model(detector, model, adv_inputs_pgd, spatial_transform)		
            class_pred_roa, type_pred_roa = detector_and_model(detector, model, adv_inputs_roa, spatial_transform)			
            class_pred_fra, type_pred_fra = detector_and_model(detector, model, adv_inputs_fra, spatial_transform)			
            class_pred_one, type_pred_one = detector_and_model(detector, model, adv_inputs_one, spatial_transform)			
            #print(type_pred_pgd)
        if opt.save_image:		
            visual_results(adv_inputs_one, perturb_pgd, opt.sample_duration, video_path, opt)	                

        # model evaluation
        if class_pred_clean == targets:
            correct_clean = correct_clean + 1
        if class_pred_pgd == targets:
            correct_pgd = correct_pgd + 1
        if class_pred_roa == targets:
            correct_roa = correct_roa + 1
        if class_pred_fra == targets:
            correct_framing = correct_framing + 1
        if class_pred_one == targets:
            correct_one = correct_one + 1
        if class_pred_clean == targets and class_pred_pgd == targets and class_pred_roa == targets and class_pred_fra == targets and class_pred_one == targets:
            correct_union = correct_union + 1 		

        acc_clean = correct_clean / count * 100
        acc_pgd = correct_pgd / count * 100
        acc_roa = correct_roa / count * 100
        acc_framing = correct_framing / count * 100
        acc_one = correct_one/ count * 100
        acc_union = correct_union/ count * 100			
		
        with open(saveDoc, "a") as myfile:
            myfile.write('id: ' + f'{id:04}' + ', acc: {0:.2f}: '.format(acc_union) + '\n')
        with open(saveDoc_clean, "a") as myfile:
            myfile.write('id: ' + f'{id:04}' + ', acc: {0:.2f}'.format(acc_clean) + '\n')
        with open(saveDoc_pgd, "a") as myfile:
            myfile.write('id: ' + f'{id:04}' + ', acc: {0:.2f}'.format(acc_pgd) + '\n')
        with open(saveDoc_roa, "a") as myfile:
            myfile.write('id: ' + f'{id:04}' + ', acc: {0:.2f}'.format(acc_roa) + '\n')
        with open(saveDoc_framing, "a") as myfile:
            myfile.write('id: ' + f'{id:04}' + ', acc: {0:.2f}'.format(acc_framing) + '\n')
        with open(saveDoc_one, "a") as myfile:
            myfile.write('id: ' + f'{id:04}' + ', acc: {0:.2f}'.format(acc_one) + '\n')                
		
        print('id: ' + f'{id:04}' + ', acc: {0:.2f}'.format(acc_union))
        
        # detector evaluation        
        if type_pred_clean == 0:
            type_correct_clean = type_correct_clean + 1
        if type_pred_pgd == 1:
            type_correct_pgd = type_correct_pgd + 1
        if type_pred_roa == 2:
            type_correct_roa = type_correct_roa + 1
        if type_pred_fra == 2:
            type_correct_framing = type_correct_framing + 1
        if type_pred_one == 1:
            type_correct_one = type_correct_one + 1

        type_acc_clean = type_correct_clean / count * 100
        type_acc_pgd = type_correct_pgd / count * 100
        type_acc_roa = type_correct_roa / count * 100
        type_acc_framing = type_correct_framing / count * 100
        type_acc_one = type_correct_one/ count * 100
			
        with open(saveDoc_clean, "a") as myfile:
            myfile.write('Detect, id: ' + f'{id:04}' + ', acc: {0:.2f}, predict: '.format(type_acc_clean) + str(type_pred_clean) + '\n')
        with open(saveDoc_pgd, "a") as myfile:
            myfile.write('Detect, id: ' + f'{id:04}' + ', acc: {0:.2f}, predict: '.format(type_acc_pgd) + str(type_pred_pgd) + '\n')
        with open(saveDoc_roa, "a") as myfile:
            myfile.write('Detect, id: ' + f'{id:04}' + ', acc: {0:.2f}, predict: '.format(type_acc_roa) + str(type_pred_roa) + '\n')
        with open(saveDoc_framing, "a") as myfile:
            myfile.write('Detect, id: ' + f'{id:04}' + ', acc: {0:.2f}, predict: '.format(type_acc_framing) + str(type_pred_fra) + '\n')
        with open(saveDoc_one, "a") as myfile:
            myfile.write('Detect, id: ' + f'{id:04}' + ', acc: {0:.2f}, predict: '.format(type_acc_one) + str(type_pred_one) + '\n')                
		
        #print('id: ' + f'{count:04}' + ', acc: {0:.2f}'.format(acc_union))		 
    
    with open(saveDoc, "a") as myfile:
        myfile.write('\n' + str(datetime.datetime.now()-time_start) + ',       ' + str(datetime.datetime.now()) + '\n')    
	#'''
    print('Clean detect: {0:.2f}: '.format(type_acc_clean))
    print('PGD detect: {0:.2f}: '.format(type_acc_pgd))
    print('ROA detect: {0:.2f}: '.format(type_acc_roa))
    print('Framing detect: {0:.2f}: '.format(type_acc_framing))
    print('One detect: {0:.2f}: '.format(type_acc_one))
    print('Avg detect: {0:.2f}: '.format((type_acc_clean+type_acc_pgd+type_acc_roa+type_acc_framing+type_acc_one) / 5))
    print('Clean acc: {0:.2f}: '.format(acc_clean))
    print('PGD acc: {0:.2f}: '.format(acc_pgd))
    print('ROA acc: {0:.2f}: '.format(acc_roa))
    print('Framing acc: {0:.2f}: '.format(acc_framing))
    print('One acc: {0:.2f}: '.format(acc_one))
    print('Avg acc: {0:.2f}: '.format((acc_clean+acc_pgd+acc_roa+acc_framing+acc_one) / 5))
    print('Union acc: {0:.2f}: '.format(acc_union))

		