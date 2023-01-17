import torch
from torch import nn

from models import resnet, resnext_3bn, resnext, resnext_oun, resnext_3bn_comb, resnext_onthefly


def generate_model(model_name, sample_duration=40):

    if model_name == 'resnet':
        model = resnet.resnet50(num_classes=3, shortcut_type='B',
                                sample_size=112, sample_duration=sample_duration,
                                last_fc=True)

    elif model_name == 'resnext_3bn':
        model = resnext_3bn.resnet101(num_classes=101, shortcut_type='B',
            cardinality=32,
            sample_size=112, sample_duration=sample_duration)

    elif model_name == 'resnext_3bn_comb':
        model = resnext_3bn_comb.resnext101_resnet50_test(sample_duration)	

    elif model_name == 'resnext_onthefly':
        model = resnext_onthefly.ResNeXt_ResNet(n_classes_1=1, n_classes_2=101, sample_duration=sample_duration)

    elif model_name == 'resnext':
        model = resnext.resnet101(
            num_classes=101,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=sample_duration)
			
    elif model_name == 'resnext_oun':
        model = resnext_oun.resnet101(
            num_classes=101,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=sample_duration)			

    #model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model

