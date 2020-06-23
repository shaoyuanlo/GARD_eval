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
from classify_union import classify_video_adv

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.sample_size = 112
    opt.sample_duration = 40
    opt.n_classes = 101
    opt.resnet_shortcut = 'B'
    opt.model_name = 'resnext_3bn'	
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
    correct_clean = 0
    correct_pgd = 0
    correct_roa = 0
    correct_framing = 0
    correct_one = 0
    correct_union = 0	
    
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
			 
            count = count + 1			
			
            label = video_path.split('/v_')[0].split('_jpg/')[1]		
		
            result_clean, result_pgd, result_roa, result_framing, result_one = classify_video_adv(video_path, input_file, class_names, model, label, opt)			
			
            predict_clean = result_clean.split(' ')[1]
            predict_pgd = result_pgd.split(' ')[1]
            predict_roa = result_roa.split(' ')[1]
            predict_framing = result_framing.split(' ')[1]
            predict_one = result_one.split(' ')[1]
            			
            if predict_clean == label:
                correct_clean = correct_clean + 1
            if predict_pgd == label:
                correct_pgd = correct_pgd + 1
            if predict_roa == label:
                correct_roa = correct_roa + 1
            if predict_framing == label:
                correct_framing = correct_framing + 1
            if predict_one == label:
                correct_one = correct_one + 1
            if predict_clean == label and predict_pgd == label and predict_roa == label and predict_framing == label and predict_one == label:
                correct_union = correct_union + 1                
                                
            acc_clean = correct_clean / count * 100
            acc_pgd = correct_pgd / count * 100
            acc_roa = correct_roa / count * 100
            acc_framing = correct_framing / count * 100
            acc_one = correct_one/ count * 100
            acc_union = correct_union/ count * 100

            with open(saveDoc, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}: '.format(acc_union) + '\n')
            with open(saveDoc_clean, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_clean) + label + ', predict: ' + predict_clean + '\n')
            with open(saveDoc_pgd, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_pgd) + label + ', predict: ' + predict_pgd + '\n')
            with open(saveDoc_roa, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_roa) + label + ', predict: ' + predict_roa + '\n')
            with open(saveDoc_framing, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_framing) + label + ', predict: ' + predict_framing + '\n')
            with open(saveDoc_one, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc_one) + label + ', predict: ' + predict_one + '\n')                
				
            print('id: ' + f'{count:04}' + ', acc: {0:.2f} '.format(acc_union))
            #print('id: ' + f'{count:04}')			

        else:
            print('{} does not exist'.format(input_file))

    with open(saveDoc, "a") as myfile:
        myfile.write('\n' + str(datetime.datetime.now()-time_start) + ',       ' + str(datetime.datetime.now()) + '\n')			

    print('Clean acc: {0:.2f}: '.format(acc_clean))
    print('PGD acc: {0:.2f}: '.format(acc_pgd))
    print('ROA acc: {0:.2f}: '.format(acc_roa))
    print('Framing acc: {0:.2f}: '.format(acc_framing))
    print('One acc: {0:.2f}: '.format(acc_one))
    print('Avg acc: {0:.2f}: '.format((acc_clean+acc_pgd+acc_roa+acc_framing+acc_one) / 5))
    print('Union acc: {0:.2f}: '.format(acc_union))
        
