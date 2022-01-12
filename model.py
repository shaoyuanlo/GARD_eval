import torch
from torch import nn

from models import resnet, resnext_3bn, resnext, resnext_oun, resnext_3bn_comb


def generate_model(model_name):

    if model_name == 'resnet':
        model = resnet.resnet50(num_classes=3, shortcut_type='B',
                                sample_size=112, sample_duration=40,
                                last_fc=True)

    elif model_name == 'resnext_3bn':
        model = resnext_3bn.resnet101(num_classes=101, shortcut_type='B',
            cardinality=32,
            sample_size=112, sample_duration=40)

    elif model_name == 'resnext_3bn_comb':
        model = resnext_3bn_comb.resnext101_resnet50_test()	

    elif model_name == 'resnext':
        model = resnext.resnet101(
            num_classes=101,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=40)
			
    elif model_name == 'resnext_oun':
        model = resnext_oun.resnet101(
            num_classes=101,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=40)			

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    return model
