import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


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


class Denoising(nn.Module):

    def __init__(self, n_in):
        super(Denoising, self).__init__()

        self.conv_theta_01 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_02 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_03 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_04 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_05 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_06 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_07 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_08 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_09 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_10 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_11 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_12 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_13 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_14 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_15 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_16 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_17 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_18 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_19 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_theta_20 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)

        self.conv_phi_01 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_02 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_03 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_04 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_05 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_06 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_07 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_08 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_09 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_10 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_11 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_12 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_13 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_14 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_15 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_16 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_17 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_18 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_19 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
        self.conv_phi_20 = nn.Conv2d(n_in, int(n_in/2), 1, bias=False)
		
        self.conv_out = nn.Conv3d(n_in, n_in, 1, bias=False)

    def forward(self, x_in):
        H, W = x_in.shape[3:]	

        x = x_in[:,:,0]			
        theta = self.conv_theta_01(x)
        phi = self.conv_phi_01(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)		
        out = f.unsqueeze(2)
        if x_in.shape[2]==1:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out		

        x = x_in[:,:,1]
        theta = self.conv_theta_02(x)
        phi = self.conv_phi_02(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==2:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out
			
        x = x_in[:,:,2]
        theta = self.conv_theta_03(x)
        phi = self.conv_phi_03(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==3:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,3]
        theta = self.conv_theta_04(x)
        phi = self.conv_phi_04(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==4:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,4]
        theta = self.conv_theta_05(x)
        phi = self.conv_phi_05(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==5:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,5]
        theta = self.conv_theta_06(x)
        phi = self.conv_phi_06(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==6:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,6]
        theta = self.conv_theta_07(x)
        phi = self.conv_phi_07(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==7:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,7]
        theta = self.conv_theta_08(x)
        phi = self.conv_phi_08(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==8:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,8]
        theta = self.conv_theta_09(x)
        phi = self.conv_phi_09(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==9:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,9]
        theta = self.conv_theta_10(x)
        phi = self.conv_phi_10(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==10:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,10]
        theta = self.conv_theta_11(x)
        phi = self.conv_phi_11(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==11:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,11]
        theta = self.conv_theta_12(x)
        phi = self.conv_phi_12(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==12:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,12]
        theta = self.conv_theta_13(x)
        phi = self.conv_phi_13(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==13:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,13]
        theta = self.conv_theta_14(x)
        phi = self.conv_phi_14(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==14:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,14]
        theta = self.conv_theta_15(x)
        phi = self.conv_phi_15(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==15:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,15]
        theta = self.conv_theta_16(x)
        phi = self.conv_phi_16(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==16:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,16]
        theta = self.conv_theta_17(x)
        phi = self.conv_phi_17(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==17:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,17]
        theta = self.conv_theta_18(x)
        phi = self.conv_phi_18(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==18:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,18]
        theta = self.conv_theta_19(x)
        phi = self.conv_phi_19(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)
        if x_in.shape[2]==19:		
            out = self.conv_out(out)	
            out = out + x_in    
            return out

        x = x_in[:,:,19]
        theta = self.conv_theta_20(x)
        phi = self.conv_phi_20(x)			    
        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        orig_shape = f.shape				
        f = torch.reshape(f, (-1, H*W, H*W))		
        f = f / torch.sqrt(torch.tensor([theta.shape[1]], dtype=torch.float32).cuda())
        f = F.softmax(f, dim=2)  # dimension!!!	
        f = torch.reshape(f, orig_shape)		
        f = torch.einsum('nabcd,nicd->niab', f, x)
        out = torch.cat([out, f.unsqueeze(2)], dim=2)		
        out = self.conv_out(out)	
        out = out + x_in    
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.denoise_res2 = Denoising(64)
        self.denoise_res3 = Denoising(128)
        self.denoise_res4 = Denoising(256)
        self.denoise_res5 = Denoising(512)

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

        return nn.Sequential(*layers)

    def forward(self, x):	
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)		
        x = self.maxpool(x)

        x = self.layer1(x)  # res2		
        x = self.denoise_res2(x)
		
        x = self.layer2(x)  # res3
        x = self.denoise_res3(x)
		
        x = self.layer3(x)  # res4
        x = self.denoise_res4(x)
		
        x = self.layer4(x)  # res5
        x = self.denoise_res5(x)	

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


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
