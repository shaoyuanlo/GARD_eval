import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3),out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='A', num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
		
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
		
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
		
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mode=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_f = self.layer1(x)
        x = self.layer2(x_f)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
			

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
		
        mid_planes = cardinality * int(planes / 32)
		
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.bn1_b = nn.BatchNorm3d(mid_planes)
        self.bn1_c = nn.BatchNorm3d(mid_planes)	
		
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.bn2_b = nn.BatchNorm3d(mid_planes)
        self.bn2_c = nn.BatchNorm3d(mid_planes)
		
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.bn3_b = nn.BatchNorm3d(planes * self.expansion)
        self.bn3_c = nn.BatchNorm3d(planes * self.expansion)
		
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, prob):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out) * prob[:,0].view(-1,1,1,1,1) + self.bn1_b(out) * prob[:,1].view(-1,1,1,1,1) + self.bn1_c(out) * prob[:,2].view(-1,1,1,1,1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) * prob[:,0].view(-1,1,1,1,1) + self.bn2_b(out) * prob[:,1].view(-1,1,1,1,1) + self.bn2_c(out) * prob[:,2].view(-1,1,1,1,1)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) * prob[:,0].view(-1,1,1,1,1) + self.bn3_b(out) * prob[:,1].view(-1,1,1,1,1) + self.bn3_c(out) * prob[:,2].view(-1,1,1,1,1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', cardinality=32, num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
		
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.bn1_b = nn.BatchNorm3d(64)
        self.bn1_c = nn.BatchNorm3d(64)		
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
		
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
		
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return MultiPrmSequential(*layers)

    def forward(self, x, prob):
	
        x = self.conv1(x)
        x = self.bn1(x) * prob[:,0].view(-1,1,1,1,1) + self.bn1_b(x) * prob[:,1].view(-1,1,1,1,1) + self.bn1_c(x) * prob[:,2].view(-1,1,1,1,1)
		
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, prob)
        x = self.layer2(x, prob)
        x = self.layer3(x, prob)
        x = self.layer4(x, prob)

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


class ResNeXt_ResNet(nn.Module):

    def __init__(self, sample_size=112, sample_duration=40, shortcut_type='B', cardinality=32, n_classes_1=400, n_classes_2=400):
        super(ResNeXt_ResNet, self).__init__()

        # Detector: ResNet-18		
        self.resnet = ResNet(BasicBlock, [2,2,2,2], sample_size, sample_duration, shortcut_type='A', num_classes=n_classes_1)
        # Classifier: ResNeXt-101		
        self.resnext = ResNeXt(ResNeXtBottleneck, [3,4,23,3], sample_size, sample_duration, shortcut_type, cardinality, num_classes=n_classes_2)

    def forward(self, x, mode=None):

        detect = self.resnet(x)
		
        #detect_gumbel = F.gumbel_softmax(detect, dim=1)		
        detect_gumbel = F.softmax(detect, dim=1)		
		
        out = self.resnext(x, detect_gumbel)

        if mode=='train':
            return out, detect
        else:
            return out


def resnext101_resnet50():
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt_ResNet()
	
    return model


def resnext101_resnet50_test():
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt_ResNet(n_classes_1=3, n_classes_2=101)
	
    return model

	