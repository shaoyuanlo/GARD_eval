import torch
from torch import nn

from models import resnet, resnext, resnext_3bn, resnext_denoise, wide_resnet
from models import resnet_oucd, resnet_denoise, resnet_denoise2d, resnet_oucd_loss, resnet_oun, resnet_oun_bn, resnet_un, resnet_on


def generate_model(opt):
    assert opt.model in [
        'resnet', 'wideresnet', 'wideresnet_3bn', 'resnext', 'resnext_3bn', 'resnext_denoise', 'resnet_oucd', 'resnet_denoise',
        'resnet_denoise2d', 'resnet_oucd_loss', 'resnet_oun', 'resnet_un', 'resnet_on', 'resnet_oun_bn']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
				
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        from models.wide_resnet import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
				
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
				
    elif opt.model == 'resnext_3bn':
        assert opt.model_depth in [101]

        from models.resnext_3bn import get_fine_tuning_parameters

        model = resnext_3bn.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'resnext_denoise':
        assert opt.model_depth in [101]

        from models.resnext_denoise import get_fine_tuning_parameters

        model = resnext_denoise.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'resnet_denoise':

        from models.resnet_denoise import get_fine_tuning_parameters

        model = resnet_denoise.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
			
    elif opt.model == 'resnet_denoise2d':

        from models.resnet_denoise2d import get_fine_tuning_parameters

        model = resnet_denoise2d.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)			

    elif opt.model == 'resnet_oucd':

        from models.resnet_oucd import get_fine_tuning_parameters

        model = resnet_oucd.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'resnet_oucd_loss':

        from models.resnet_oucd_loss import get_fine_tuning_parameters

        model = resnet_oucd_loss.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'resnet_un':

        from models.resnet_un import get_fine_tuning_parameters

        model = resnet_un.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)			

    elif opt.model == 'resnet_on':

        from models.resnet_on import get_fine_tuning_parameters

        model = resnet_on.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
			
    elif opt.model == 'resnet_oun':

        from models.resnet_oun import get_fine_tuning_parameters

        model = resnet_oun.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)			

    elif opt.model == 'resnet_oun_bn':

        from models.resnet_oun_bn import get_fine_tuning_parameters

        model = resnet_oun_bn.resnet18(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
			
    elif opt.model == 'wideresnet_3bn':
        assert opt.model_depth in [50]

        from models.wide_resnet_3bn import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet_3bn.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)			


    # === Get Model Parameters === #
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    if opt.pretrain_path=='/home/sylo/SegNet/flowattack/3D-ResNets-PyTorch/none':
        return model, model.parameters()

    else:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)

        if opt.model=='resnet' or opt.model=='resnext' or opt.model=='wideresnet':
            assert opt.arch == pretrain['arch']				
            model.load_state_dict(pretrain['state_dict'])

        elif opt.model=='resnext_3bn' or opt.model=='wideresnet_3bn':			
            model_dict = model.state_dict()
            k_list = []
            count = -1	               
            for k, v in model_dict.items():
                conut = count + 1                
                k_list.append(k)					
                flag = 0				
                for k2, v2 in pretrain['state_dict'].items():
                    if k==k2: 
                        model_dict[k] = v2
                        flag = 1							
                        break
                if flag==0 and ('num_batches_tracked' not in k):                                          
                    model_dict[k] = model_dict[k_list[count-5]]					
            model.load_state_dict(model_dict)
								
        else:		
            model_dict = model.state_dict()	               
            for k, v in model_dict.items():                				
                for k2, v2 in pretrain['state_dict'].items():
                    if k==k2: 
                        model_dict[k] = v2							
                        break				
            model.load_state_dict(model_dict)

        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()
		
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
		
        return model, parameters

