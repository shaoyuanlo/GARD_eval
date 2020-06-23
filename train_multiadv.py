import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy
from video_classification.advertorch.attacks.my_videoattack import LinfPGDAttack
from video_classification.advertorch.attacks.my_videoattack_bn import LinfPGDAttack_bn
from video_classification.sticker_attack import ROA


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
	
    if epoch<=1:
        print('using avg')
    else:
        print('using max')	

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

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
        
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if i>=3:
            break
		
        if i>=2384:  # for batch_size 4, 1589 for batch_size 6, 1192 for batch_size 8
            print('inputs.shape[0] != targets.shape[0]')
            break

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        
        model.eval()

        # set attack type number        
        type_train = -1        
        type_clean = 0
        type_pgd = 1
        type_roa = 2		
		
        # ROA attack
        if opt.model=='resnext_3bn' or opt.model=='wideresnet_3bn':        
            adversary_roa = ROA(model, opt.sample_size)
            adv_inputs_roa, perturb_roa = adversary_roa.random_search_bn(inputs, type_roa, targets, opt.sample_duration, opt.step_size, 
                                opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)   # ROA type=2    
        else:
            adversary_roa = ROA(model, opt.sample_size)
            adv_inputs_roa, perturb_roa = adversary_roa.random_search(inputs, targets, opt.sample_duration, opt.step_size, 
                                opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)      

        #adversary = ROA(model, opt.sample_size)
        #adv_inputs, perturb = adversary.random_search_one(inputs, targets, opt.num_pixel, opt.sparsity, opt.step_size, opt.attack_iter)								

        # Noise attack
        if opt.model=='resnext_3bn' or opt.model=='wideresnet_3bn':		
            adversary_pgd = LinfPGDAttack_bn(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
            adv_inputs_pgd, perturb_pgd = adversary_pgd.perturb(inputs, type_pgd, attack_mask, targets)  # PGD type=1
        else:
            adversary_pgd = LinfPGDAttack(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=opt.step_size)
            adv_inputs_pgd, perturb_pgd = adversary_pgd.perturb(inputs, attack_mask, targets)
		
        #'''
        #if epoch<=1:        
        model.train()	        
        adv_inputs = torch.cat((inputs, adv_inputs_pgd, adv_inputs_roa), dim=0)
        if opt.model=='resnext_3bn' or opt.model=='wideresnet_3bn':	        
            outputs = model(adv_inputs, type_train)
        else:		
            outputs = model(adv_inputs)            
        
        targets_cat = torch.cat((targets, targets, targets), dim=0)
        #'''			
        '''
        else:		
            if opt.model=='resnext_3bn' or opt.model=='wideresnext_3bn':	        
                outputs_pgd = model(adv_inputs_pgd, 1)
                outputs = model(adv_inputs_roa, 2)
            else:		
                outputs_pgd = model(adv_inputs_pgd)
                outputs = model(adv_inputs_roa)		  
            
            loss_pgd = criterion(outputs_pgd, targets)		
            loss = criterion(outputs, targets)
            if loss_pgd < loss:
                adv_inputs_pgd = adv_inputs_roa
                type_train = 22
            else:
                type_train = 11		
            
            model.train()			
            adv_inputs_pgd = torch.cat((inputs, adv_inputs_pgd), dim=0)
            if opt.model=='resnext_3bn' or opt.model=='wideresnext_3bn':	        
                outputs = model(adv_inputs_pgd, type_train)
            else:		
                outputs = model(adv_inputs_pgd)		
		    
            targets_cat = torch.cat((targets, targets), dim=0)			
        '''		
		
        loss = criterion(outputs, targets_cat)		
		
        output_clean = torch.split(outputs, opt.batch_size, dim=0)	
        output_clean = output_clean[0]
        acc = calculate_accuracy(output_clean, targets)		
        #losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.data, inputs.size(0))		
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
