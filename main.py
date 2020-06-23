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
from classify import classify_video, classify_video_adv

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.sample_size = 112
    opt.sample_duration = 40
    opt.n_classes = 101
    opt.model_depth	= 101
    opt.resnet_shortcut = 'B'	
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

    #if os.path.exists('tmp'):
        #subprocess.call('rm -rf tmp', shell=True)

	# === save information in .txt files === #
    savePath = 'results/'	
    saveDoc = savePath + opt.exp_name + '.txt' 
    print('=> will save everything to {}'.format(saveDoc))	 
    with open(saveDoc, "a") as myfile:
        myfile.writelines(str(opt) + '\n\n')	
        LL = ['-input: '+str(opt.input)+'\n','-model: '+str(opt.model)+'\n\n']	
        myfile.writelines(LL)
        time_start = datetime.datetime.now()
        myfile.write(str(time_start) + '\n\n')		
		
    #outputs = []
    count = 0
    correct = 0	
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            #print(video_path)
            #subprocess.call('mkdir tmp', shell=True)
            #subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path), shell=True)
			 
            count = count + 1			
			
            label = video_path.split('/v_')[0].split('_jpg/')[1]		

            if opt.attack_type == 'clean':			
                result = classify_video(video_path, input_file, class_names, model, opt)
            else:				
                result = classify_video_adv(video_path, input_file, class_names, model, label, opt)			
			
            predict = result.split(' ')[1]
			
            if predict == label:
                correct = correct + 1
            acc = correct / count * 100

            with open(saveDoc, "a") as myfile:
                myfile.write('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc) + label + ', predict: ' + predict + '\n')				
            print('id: ' + f'{count:04}' + ', acc: {0:.2f}, label: '.format(acc) + label + ', predict: ' + predict)				
			
            #outputs.append(result)

            #subprocess.call('rm -rf tmp', shell=True)
        else:
            print('{} does not exist'.format(input_file))

    with open(saveDoc, "a") as myfile:
        myfile.write('\n' + str(datetime.datetime.now()-time_start) + ',       ' + str(datetime.datetime.now()) + '\n')			
			
    #if os.path.exists('tmp'):
        #subprocess.call('rm -rf tmp', shell=True)

    #with open(opt.output, 'w') as f:
        #json.dump(outputs, f)
