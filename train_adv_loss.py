import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy
from video_classification.advertorch.attacks.my_videoattack import LinfPGDAttack, L2PGDAttack
from video_classification.sticker_attack import ROA


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    criterion_aux = nn.MSELoss()

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
        #opt.epsilon = 255
        
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
        elif opt.attack_type == 'noise':
            adversary = LinfPGDAttack(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=float(opt.step_size/255))
            adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)			
        elif opt.attack_type == 'pgd_l2':
            adversary = L2PGDAttack(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon*40), nb_iter=opt.attack_iter, eps_iter=float(opt.step_size))
            adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)

        model_copy = model
        model_copy.eval()		
        model.train()

        _, out_res2, _, out_res3, _, out_res4, _, out_res5, _ = model_copy(inputs, mode='multi')
        outputs, _, out_de_res2, _, out_de_res3, _, out_de_res4, _, out_de_res5 = model(adv_inputs, mode='multi')		
		
        loss_res2 = criterion_aux(out_de_res2, out_res2)
        loss_res3 = criterion_aux(out_de_res3, out_res3)
        loss_res4 = criterion_aux(out_de_res4, out_res4)
        loss_res5 = criterion_aux(out_de_res5, out_res5)		
        loss = criterion(outputs, targets)
        #loss = loss*0.5 + loss_res2*0.125 + loss_res3*0.125 + loss_res4*0.125 + loss_res5*0.125	
        loss = loss*0.8 + loss_res2*0.05 + loss_res3*0.05 + loss_res4*0.05 + loss_res5*0.05

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
	
