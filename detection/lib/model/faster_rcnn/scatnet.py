from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb


__all__ = ['ScatNet']

model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}





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


        self.avgpool = nn.AvgPool2d((self.nspace+2**(len(channels)-2))//(2**(len(channels)-1)), stride=1)
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






def new_scat_bottleneck(N,J,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ScatResNet(J,N,Bottleneck,kwargs['width'],kwargs['depth'],kwargs['conv1x1'])

    return model




class scatnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = "./data/pretrained_model/scatresnet50.tar.gz"
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = new_scat_bottleneck(1200,3,width=[128,256,512],depth=[5,8,3],conv1x1=False)#gene_scat_basicblock(1200,3,[128,256,512],[10,10,3])
    loading_resnet = nn.DataParallel(resnet)

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path,map_location=lambda storage, location: storage)
      state_dict = state_dict['state_dict']
      #print({k for k,v in state_dict.items()})
      #print({k:v for k,v in state_dict.items() if k in loading_resnet.state_dict()})
      loading_resnet.load_state_dict({k:v for k,v in state_dict.items() if k in loading_resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.layers[0],resnet.layers[1])

    self.RCNN_top = nn.Sequential(resnet.layers[2])

    N = 2048

    self.RCNN_cls_score = nn.Linear(N, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(N, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(N, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 3)

    #if cfg.RESNET.FIXED_BLOCKS >= 2:
    #for p in self.RCNN_base[4].parameters(): p.requires_grad=False
    #if cfg.RESNET.FIXED_BLOCKS >= 1:
    #for p in self.RCNN_base[3].parameters(): p.requires_grad=False
    for p in self.RCNN_base[3].parameters(): p.requires_grad = False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)

    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[4].train()


      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)



  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
