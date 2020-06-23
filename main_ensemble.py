import os
import sys
import json
import subprocess
import datetime
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify_ensemble import classify_video_adv

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.sample_size = 112
    opt.sample_duration = 40
    opt.n_classes = 101
    opt.resnet_shortcut = 'B'
    opt.model_name = 'wideresnet_3bn'  # 'resnext_3bn'
    opt.model_depth = 50  # 101	
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
	
    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

	# === save information in .txt files === #
    savePath = 'results_union/'	
    saveDoc = savePath + opt.exp_name + '_union.txt'
    saveDoc_clean = savePath + opt.exp_name + '_clean.txt' 
    saveDoc_pgd = savePath + opt.exp_name + '_pgd.txt' 
    saveDoc_roa = savePath + opt.exp_name + '_roa.txt' 
    saveDoc_framing = savePath + opt.exp_name + '_framing.txt' 
    saveDoc_one = savePath + opt.exp_name + '_one.txt' 
    print('=> will save everything to {}'.format(saveDoc))	 
    with open(saveDoc, "a") as myfile:
        myfile.writelines(str(opt) + '\n\n')	
        LL = ['-input: '+str(opt.input)+'\n','-model: '+str(opt.model)+'\n\n']	
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
		
    count = 0
    correct_clean0 = 0
    correct_clean1 = 0
    correct_clean2 = 0	
    correct_pgd0 = 0
    correct_pgd1 = 0
    correct_pgd2 = 0	
    correct_roa0 = 0
    correct_roa1 = 0
    correct_roa2 = 0	
    correct_framing0 = 0
    correct_framing1 = 0
    correct_framing2 = 0	
    correct_one0 = 0
    correct_one1 = 0
    correct_one2 = 0	
    
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
			 
            count = count + 1			
			
            label = video_path.split('/v_')[0].split('_jpg/')[1]		
		
            result_clean0, result_pgd0, result_roa0, result_framing0, result_one0, result_clean1, result_pgd1, result_roa1, result_framing1, result_one1, result_clean2, result_pgd2, result_roa2, result_framing2, result_one2, = classify_video_adv(video_path, input_file, class_names, model, label, opt)			
			
            predict_clean0 = result_clean0.split(' ')[1]
            predict_clean1 = result_clean1.split(' ')[1]
            predict_clean2 = result_clean2.split(' ')[1]						
            predict_pgd0 = result_pgd0.split(' ')[1]
            predict_pgd1 = result_pgd1.split(' ')[1]
            predict_pgd2 = result_pgd2.split(' ')[1]						
            predict_roa0 = result_roa0.split(' ')[1]
            predict_roa1 = result_roa1.split(' ')[1]
            predict_roa2 = result_roa2.split(' ')[1]						
            predict_framing0 = result_framing0.split(' ')[1]
            predict_framing1 = result_framing1.split(' ')[1]
            predict_framing2 = result_framing2.split(' ')[1]						
            predict_one0 = result_one0.split(' ')[1]
            predict_one1 = result_one1.split(' ')[1]			
            predict_one2 = result_one2.split(' ')[1]			
            			
            if predict_clean0 == label:
                correct_clean0 = correct_clean0 + 1
            if predict_clean1 == label:
                correct_clean1 = correct_clean1 + 1
            if predict_clean2 == label:
                correct_clean2 = correct_clean2 + 1				
            if predict_pgd0 == label:
                correct_pgd0 = correct_pgd0 + 1
            if predict_pgd1 == label:
                correct_pgd1 = correct_pgd1 + 1
            if predict_pgd2 == label:
                correct_pgd2 = correct_pgd2 + 1								
            if predict_roa0 == label:
                correct_roa0 = correct_roa0 + 1
            if predict_roa1 == label:
                correct_roa1 = correct_roa1 + 1
            if predict_roa2 == label:
                correct_roa2 = correct_roa2 + 1								
            if predict_framing0 == label:
                correct_framing0 = correct_framing0 + 1
            if predict_framing1 == label:
                correct_framing1 = correct_framing1 + 1
            if predict_framing2 == label:
                correct_framing2 = correct_framing2 + 1								
            if predict_one0 == label:
                correct_one0 = correct_one0 + 1
            if predict_one1 == label:
                correct_one1 = correct_one1 + 1
            if predict_one2 == label:
                correct_one2 = correct_one2 + 1				
               
                                
            acc_clean0 = correct_clean0 / count * 100
            acc_clean1 = correct_clean1 / count * 100
            acc_clean2 = correct_clean2 / count * 100			
            acc_pgd0 = correct_pgd0 / count * 100
            acc_pgd1 = correct_pgd1 / count * 100
            acc_pgd2 = correct_pgd2 / count * 100			
            acc_roa0 = correct_roa0 / count * 100
            acc_roa1 = correct_roa1 / count * 100
            acc_roa2 = correct_roa2 / count * 100			
            acc_framing0 = correct_framing0 / count * 100
            acc_framing1 = correct_framing1 / count * 100
            acc_framing2 = correct_framing2 / count * 100			
            acc_one0 = correct_one0 / count * 100
            acc_one1 = correct_one1 / count * 100
            acc_one2 = correct_one2 / count * 100			

            with open(saveDoc, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + '\n')
            with open(saveDoc_clean, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_clean0) + label + ', predict: ' + predict_clean0 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_clean1) + ', predict: ' + predict_clean1 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_clean2) + ', predict: ' + predict_clean2 + '\n')				
            with open(saveDoc_pgd, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_pgd0) + label + ', predict: ' + predict_pgd0 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_pgd1) + ', predict: ' + predict_pgd1 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_pgd2) + ', predict: ' + predict_pgd2 + '\n')				
            with open(saveDoc_roa, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_roa0) + label + ', predict: ' + predict_roa0 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_roa1) + ', predict: ' + predict_roa1 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_roa2) + ', predict: ' + predict_roa2 + '\n')				
            with open(saveDoc_framing, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_framing0) + label + ', predict: ' + predict_framing0 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_framing1) + ', predict: ' + predict_framing1 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_framing2) + ', predict: ' + predict_framing2 + '\n')				
            with open(saveDoc_one, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_one0) + label + ', predict: ' + predict_one0 + '\n')
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_one1) + ', predict: ' + predict_one1 + '\n') 
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_one2) + ', predict: ' + predict_one2 + '\n')                 
				
            print('id: ' + f'{count:04}')
            print('Clean acc: {0:.2f}: '.format(acc_clean0))			
            print('Clean acc: {0:.2f}: '.format(acc_clean1))
            print('Clean acc: {0:.2f}: '.format(acc_clean2))			
            print('PGD acc: {0:.2f}: '.format(acc_pgd0))
            print('PGD acc: {0:.2f}: '.format(acc_pgd1))
            print('PGD acc: {0:.2f}: '.format(acc_pgd2))			
            print('ROA acc: {0:.2f}: '.format(acc_roa0))
            print('ROA acc: {0:.2f}: '.format(acc_roa1))
            print('ROA acc: {0:.2f}: '.format(acc_roa2))			
            print('Framing acc: {0:.2f}: '.format(acc_framing0))
            print('Framing acc: {0:.2f}: '.format(acc_framing1))
            print('Framing acc: {0:.2f}: '.format(acc_framing2))			
            print('One acc: {0:.2f}: '.format(acc_one0))
            print('One acc: {0:.2f}: '.format(acc_one1))
            print('One acc: {0:.2f}: '.format(acc_one2))			

        else:
            print('{} does not exist'.format(input_file))

    with open(saveDoc, "a") as myfile:
        myfile.write('\n' + str(datetime.datetime.now()-time_start) + ',       ' + str(datetime.datetime.now()) + '\n')			
        
