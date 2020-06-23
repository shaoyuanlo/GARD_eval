import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import sys

from utils import AverageMeter, calculate_accuracy
from video_classification.advertorch.attacks.my_videoattack import LinfPGDAttack, L2PGDAttack
from video_classification.sticker_attack import ROA


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

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
		
        if i>=946:  # 946 for batch_size 4, 473 for batch_size 8
            print('inputs.shape[0] != targets.shape[0]')
            break		

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
		
        if opt.attack_type == 'roa':
            adversary = ROA(model, opt.sample_size)
            adv_inputs, perturb = adversary.random_search(inputs, targets, opt.sample_duration, float(opt.step_size/255), 
                                opt.attack_iter, opt.roa_size, opt.roa_size, opt.roa_stride, opt.roa_stride)
        elif opt.attack_type == 'one':
            adversary = ROA(model, opt.sample_size)
            adv_inputs, perturb = adversary.random_search_one(inputs, targets, opt.num_pixel, opt.sparsity, float(opt.step_size/255), opt.attack_iter)								
        elif opt.attack_type == 'noise':
            adversary = LinfPGDAttack(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon/255), nb_iter=opt.attack_iter, eps_iter=float(opt.step_size/255))
            adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)			
        elif opt.attack_type == 'pgd_l2':
            adversary = L2PGDAttack(predict=model, loss_fn=criterion, 
		                       eps=float(opt.epsilon*40), nb_iter=opt.attack_iter, eps_iter=float(opt.step_size))
            adv_inputs, perturb = adversary.perturb(inputs, attack_mask, targets)	
		
        outputs = model(adv_inputs)
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
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg, accuracies.avg
