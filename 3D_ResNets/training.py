import torch
import time
import os
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy

# from advertorch.attacks import LinfPGDAttack, L2PGDAttack
import numpy as np

def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False,
                attack_type="clean",
                eps=4,
                step_size=1,
                attack_iter=5,
                use_ape=False,
                D=None):
    
    
    print('train at epoch {}'.format(epoch))
    
    eps_rvs = None
    if isinstance(eps, (list, np.ndarray)): 
        eps_rvs = eps
        
    if use_ape:
        model[0].train() #model[0].eval()
        model[1].train()
        D.train()
        lr = 0.0002#0.0002
        opt_G = torch.optim.Adam(model[0].parameters(), lr=lr, betas=(0.5, 0.999))
#         opt_G = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        loss_bce_G = torch.nn.BCELoss(reduction='sum').cuda(device)
        loss_mse = torch.nn.MSELoss().cuda(device)
        opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
        loss_bce = torch.nn.BCELoss().cuda(device)
        
    else:
        model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        
        if eps_rvs is not None:
            eps = eps_rvs[i]
        
        data_time.update(time.time() - end_time)
        inputs = inputs.to(device)
#         inputs_min, inputs_max = torch.min(inputs), torch.max(inputs)
#         inputs = ((inputs - inputs_min) / (inputs_max - inputs_min))
        targets = targets.to(device, non_blocking=True)
        
        if attack_type == "clean":
            outputs = model(inputs)
        elif attack_type == "pgd_inf":
            adversary = LinfPGDAttack(predict=model, loss_fn=criterion,
                                         eps=float(eps/255), nb_iter=attack_iter, eps_iter=float(step_size/255))
            adv_inputs = adversary.perturb(inputs, targets)
            
            if use_ape: 
                eps1, eps2 = 0.7, 0.3
                current_size = adv_inputs.size(0)
                # Train D
                t_real = torch.autograd.Variable(torch.ones(current_size).to(device))#.to(f'cuda:{model.device_ids[0]}'))
                t_fake = torch.autograd.Variable(torch.zeros(current_size).to(device))#.to(f'cuda:{model.device_ids[0]}'))

                y_real = D(adv_inputs).squeeze()
                inputs_fake = model[0](adv_inputs)
                
                y_fake = D(inputs_fake).squeeze()
                loss_D = loss_bce(y_real, t_real) + loss_bce(y_fake, t_fake)
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                
                # Train G
                for _ in range(2):
                    
                    inputs_fake = model[0](adv_inputs)
                    y_fake = D(inputs_fake).squeeze()
                    ## L1, SSIM
                    loss_G = eps1 * loss_mse(inputs_fake, inputs) + eps2 * loss_bce(y_fake, t_real) 
                    opt_G.zero_grad()
                    loss_G.backward()
                    opt_G.step()
#                     outputs = model(adv_inputs)
#                     loss = criterion(outputs, targets)
#                     loss += loss_G
#                     opt_G.zero_grad()
#                     loss.backward()
#                     opt_G.step()
#             else:
            outputs = model(adv_inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#             outputs = model(adv_inputs)
        
#         loss = criterion(outputs, targets)
            
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if use_ape:
#             loss_G = loss_bce_G(x_fake, inputs)
#             opt_G.zero_grad()
#             loss_G.backward()
#             opt_G.step()


        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         acc=accuracies))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)
