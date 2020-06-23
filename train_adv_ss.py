import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy
from video_classification.advertorch.attacks.my_videoattack import LinfPGDAttack
from video_classification.sticker_attack import ROA


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

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
		
        #if i>=3:
        #    break		
        
        if i>=2384:  # 2384 for batch_size 4, 1589 for batch_size 6, 1192 for batch_size 8
            print('inputs.shape[0] != targets.shape[0]')
            break

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        
        model.eval()
        if opt.attack_type == 'roa':
            adversary = ROA(model, opt.sample_size)
            adv_inputs, perturb = adversary.random_search(inputs, targets, opt.sparsity, alpha=float(opt.step_size/255), num_iter=opt.attack_iter,
		        width=opt.roa_size, height=opt.roa_size, xskip=opt.roa_stride, yskip=opt.roa_stride)
        else:
            adversary = LinfPGDAttack(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=float(opt.step_size/255))
            adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)			

        # Self-supervised
        curr_batch_size = targets.shape[0]
        #rotate = torch.cat((torch.zeros(curr_batch_size), torch.ones(curr_batch_size), 2*torch.ones(curr_batch_size), 3*torch.ones(curr_batch_size)), dim=0)
        #rotate = rotate.long().cuda()		
        #adv_inputs = torch.cat((adv_inputs, adv_inputs, adv_inputs.transpose(3,4), adv_inputs.flip(3), adv_inputs.transpose(3,4).flip(4)), dim=0)

        rotate = torch.cat((torch.zeros(curr_batch_size), torch.ones(curr_batch_size), torch.ones(curr_batch_size), torch.ones(curr_batch_size)), dim=0)
        rotate = rotate.long().cuda()
        rand_1 = torch.randperm(40)
        rand_2 = torch.randperm(40)
        rand_3 = torch.randperm(40)		
        adv_inputs = torch.cat((adv_inputs, adv_inputs, adv_inputs[:,:,rand_1], adv_inputs[:,:,rand_2], adv_inputs[:,:,rand_3]), dim=0)		
        # ===

        model.train()		
        outputs, outputs_ss = model(adv_inputs, curr_batch_size, mode='ss')	
        outputs_ss = model.rot_pred(outputs_ss)
		
        loss = criterion(outputs, targets)
        loss_ss = criterion(outputs_ss, rotate)
        loss = loss*0.67 + loss_ss*0.33

        acc = calculate_accuracy(outputs, targets)		

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

    #save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'arch': opt.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    #torch.save(states, save_file_path)

    return states
	
