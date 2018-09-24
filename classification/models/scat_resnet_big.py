from __future__ import division
import math

import torch.nn as nn



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes,  kernel_size=1, bias=False)

class conv1x1_layer(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(conv1x1_layer, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

class LinScat(nn.Module):
    def __init__(self, J=3, N=224, num_classes=1000):
        super(LinScat, self).__init__()
        self.nspace = N // (2 ** J)
        self.nfscat = (1 + 8 * J)*3
        self.lin =  nn.Linear(self.nspace*self.nfscat*self.nspace, num_classes)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        print(x.size())
        print(self.lin)
        return self.lin(x)



class ScatResNet(nn.Module):

    def __init__(self, J,N,block, channels,layers,use_conv1x1=0,num_classes=1000):
        self.inplanes = channels[0]
        super(ScatResNet, self).__init__()

        self.nspace = N // (2 ** J)
        self.nfscat = (1 + 8 * J)

        self.block_of_1x1=False

        if use_conv1x1 == 0:
            self.conv1 = nn.Conv2d(3 * self.nfscat, channels[0], kernel_size=3, stride=1, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
        else:
            self.conv1 = nn.Conv2d(3 * self.nfscat, channels[0]*block.expansion, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0]*block.expansion)
            if use_conv1x1>1:
                self.block_of_1x1 = True
                self.conv1block = []
                for i in range(1,use_conv1x1):
                    self.conv1block.append(conv1x1_layer(channels[0]*block.expansion,channels[0]*block.expansion))
                self.conv1block = nn.ModuleList(self.conv1block)
            self.inplanes = channels[0]*block.expansion

        self.relu = nn.ReLU(inplace=True)

        self.layers=[]
        self.layers.append( self._make_layer(block, channels[0], layers[0]))

        for i in range(1,len(channels)):
            self.layers.append(self._make_layer(block, channels[i], layers[i], stride=2))
        self.layers = nn.ModuleList(self.layers)


        self.avgpool = nn.AvgPool2d(int((self.nspace+2**(len(channels)-2))//(2**(len(channels)-1))), stride=1)
        print(channels)
        print(len(channels))
        self.fc = nn.Linear(channels[len(channels)-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 3 * self.nfscat, self.nspace, self.nspace)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.block_of_1x1:
            for i in range(len(self.conv1block)):
                x = self.conv1block[i](x)

        for i in range(len(self.layers)):
            x = self.layers[i](x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def scat50(N,J,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ScatResNet(J,N,Bottleneck, [128, 256, 512],[5, 8, 3])

    return model


def wide_scat50(N,J,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ScatResNet(J,N,Bottleneck, [256, 512, 1024],[5, 8, 3])

    return model

def gene_scat50(N,J,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ScatResNet(J,N,Bottleneck,kwargs['width'],kwargs['depth'])

    return model




def gene_scat_basicblock(N,J,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ScatResNet(J,N,BasicBlock,kwargs['width'],kwargs['depth'])

    return model



def scat_lin(N, J,**kwargs):
    model = LinScat(J,N)
    return model




def scatresnet6_2(N,J):

    """Constructs a Scatter + ResNet-10 model.
    Args:
        N: is the crop size (normally 224)
	J: scattering scale (normally 3,4,or 5 for imagenet)
    """
    model = ScatResNet_old(J,N)
    return model


class ScatResNet_old(nn.Module):
    def __init__(self, J=3, N=224, num_classes=1000):
        super(ScatResNet_old, self).__init__()

        print(J)
        self.nspace = N / (2 ** J)
        self.nfscat = (1 + 8 * J )
        self.ichannels = 256
        self.ichannels2 = 512
        self.inplanes = self.ichannels
        print(self.nfscat)
        print(self.nspace)
        self.bn0 = nn.BatchNorm2d(3 * self.nfscat, eps=1e-5, momentum=0.9, affine=False)
        self.conv1 = nn.Conv2d(3 * self.nfscat, self.ichannels, kernel_size=3,
                               padding=1)  # conv3x3_3D(self.nfscat,self.ichannels)
        self.bn1 = nn.BatchNorm2d(self.ichannels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, self.ichannels, 2)
        self.layer2 = self._make_layer(BasicBlock, self.ichannels2, 2, stride=2)
        self.avgpool = nn.AvgPool2d(self.nspace / 2)

        self.fc = nn.Linear(self.ichannels2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if (m.affine):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 3 * self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)
        #  x = x.view(x.size(0), 3,self.nfscat, self.nspace, self.nspace)
        # x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.view(x.size(0), self.ichannels, self.nspace, self.nspace)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x