import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if i>=2384:  # 2384 for batch_size 4, 1589 for batch_size 6, 1192 for batch_size 8
            print('inputs.shape[0] != targets.shape[0]')
            break		
		
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)

        # Self-supervised
        curr_batch_size = targets.shape[0]
        #rotate = torch.cat((torch.zeros(curr_batch_size), torch.ones(curr_batch_size), 2*torch.ones(curr_batch_size), 3*torch.ones(curr_batch_size)), dim=0)
        #rotate = rotate.long().cuda()		
        #inputs = torch.cat((inputs, inputs, inputs.transpose(3,4), inputs.flip(3), inputs.transpose(3,4).flip(4)), dim=0)

        rotate = torch.cat((torch.zeros(curr_batch_size), torch.ones(curr_batch_size), torch.ones(curr_batch_size), torch.ones(curr_batch_size)), dim=0)
        rotate = rotate.long().cuda()
        rand_1 = torch.randperm(40)
        rand_2 = torch.randperm(40)
        rand_3 = torch.randperm(40)
        input_1 = inputs[:,:,rand_1]
        input_2 = inputs[:,:,rand_2]
        input_3 = inputs[:,:,rand_3]		
        inputs = torch.cat((inputs, inputs, input_1, input_2, input_3), dim=0)		
        # ===
		
        outputs, outputs_ss = model(inputs, curr_batch_size, mode='ss')	
        outputs_ss = model.rot_pred(outputs_ss)
		
        loss = criterion(outputs, targets)
        loss_ss = criterion(outputs_ss, rotate)
        loss = loss*0.67 + loss_ss*0.33		
		
        acc = calculate_accuracy(outputs, targets)		
        #losses.update(loss.data[0], inputs.size(0))
        losses.update(loss, inputs.size(0))		
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

