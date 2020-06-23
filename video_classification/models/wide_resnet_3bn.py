import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['WideResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class WideBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.bn1_b = nn.BatchNorm3d(planes)
        self.bn1_c = nn.BatchNorm3d(planes)			
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.bn2_b = nn.BatchNorm3d(planes)
        self.bn2_c = nn.BatchNorm3d(planes)			
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.bn3_b = nn.BatchNorm3d(planes * self.expansion)
        self.bn3_c = nn.BatchNorm3d(planes * self.expansion)			
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, type):
        residual = x

        out = self.conv1(x)
		
        if type==0:
            out = self.bn1(out)
        elif type==1:
            out = self.bn1_b(out)
        elif type==2:
            out = self.bn1_c(out)
        elif type==11:
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_pgd= self.bn1_b(slice[1])		
            out = torch.cat((slice_clean,slice_pgd), dim=0)
        elif type==22:
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_roa = self.bn1_c(slice[1])
            out = torch.cat((slice_clean,slice_roa), dim=0)
        elif type==33:        
            slice = torch.split(out, 1, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_pgd = self.bn1_b(slice[1])		
            slice_roa = self.bn1_c(slice[2])
            out = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)			
        else:        
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_pgd = self.bn1_b(slice[1])		
            slice_roa = self.bn1_c(slice[2])
            out = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)
			
        out = self.relu(out)

        out = self.conv2(out)
		
        if type==0:
            out = self.bn2(out)
        elif type==1:
            out = self.bn2_b(out)
        elif type==2:
            out = self.bn2_c(out)
        elif type==11:
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn2(slice[0])		
            slice_pgd= self.bn2_b(slice[1])		
            out = torch.cat((slice_clean,slice_pgd), dim=0)
        elif type==22:
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn2(slice[0])		
            slice_roa = self.bn2_c(slice[1])
            out = torch.cat((slice_clean,slice_roa), dim=0)
        elif type==33:          
            slice = torch.split(out, 1, dim=0)		
            slice_clean = self.bn2(slice[0])		
            slice_pgd = self.bn2_b(slice[1])		
            slice_roa = self.bn2_c(slice[2])
            out = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)			
        else:          
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn2(slice[0])		
            slice_pgd= self.bn2_b(slice[1])		
            slice_roa = self.bn2_c(slice[2])
            out = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)
			
        out = self.relu(out)

        out = self.conv3(out)
		
        if type==0:
            out = self.bn3(out)
        elif type==1:
            out = self.bn3_b(out)
        elif type==2:
            out = self.bn3_c(out)
        elif type==11:
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn3(slice[0])		
            slice_pgd= self.bn3_b(slice[1])		
            out = torch.cat((slice_clean,slice_pgd), dim=0)
        elif type==22:
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn3(slice[0])		
            slice_roa = self.bn3_c(slice[1])
            out = torch.cat((slice_clean,slice_roa), dim=0)
        elif type==33:          
            slice = torch.split(out, 1, dim=0)		
            slice_clean = self.bn3(slice[0])		
            slice_pgd = self.bn3_b(slice[1])		
            slice_roa = self.bn3_c(slice[2])
            out = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)			
        else:          
            slice = torch.split(out, 4, dim=0)		
            slice_clean = self.bn3(slice[0])		
            slice_pgd = self.bn3_b(slice[1])		
            slice_roa = self.bn3_c(slice[2])
            out = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WideResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 k=1,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(WideResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.bn1_b = nn.BatchNorm3d(64)
        self.bn1_c = nn.BatchNorm3d(64)		
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * k, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128 * k, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256 * k, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512 * k, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * k * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return MultiPrmSequential(*layers)

    def forward(self, x, type):
        x = self.conv1(x)

        if type==0:
            x = self.bn1(x)
        elif type==1:
            x = self.bn1_b(x)
        elif type==2:
            x = self.bn1_c(x)
        elif type==11:
            slice = torch.split(x, 4, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_pgd= self.bn1_b(slice[1])		
            x = torch.cat((slice_clean,slice_pgd), dim=0)
        elif type==22:
            slice = torch.split(x, 4, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_roa = self.bn1_c(slice[1])
            x = torch.cat((slice_clean,slice_roa), dim=0)
        elif type==33:
            #print(x.shape)		
            slice = torch.split(x, 1, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_pgd = self.bn1_b(slice[1])		
            slice_roa = self.bn1_c(slice[2])
            x = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)			
        else:            
            slice = torch.split(x, 4, dim=0)		
            slice_clean = self.bn1(slice[0])		
            slice_pgd = self.bn1_b(slice[1])		
            slice_roa = self.bn1_c(slice[2])
            x = torch.cat((slice_clean,slice_pgd,slice_roa), dim=0)
			
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, type)
        x = self.layer2(x, type)
        x = self.layer3(x, type)
        x = self.layer4(x, type)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = WideResNet(WideBottleneck, [3, 4, 6, 3], **kwargs)
    return model

	
class MultiPrmSequential(nn.Sequential):
    def __init__(self, *args):
        super(MultiPrmSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input, type):
        for module in self:
            input = module(input, type)
        return input

		