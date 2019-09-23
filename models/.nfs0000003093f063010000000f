import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from  MPNCOV import MPNCOV
import torch.nn.functional as F
import resnet
import densenet
import senet

import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['d2ve']

class Model(nn.Module):
    def __init__(self, pretrained=True, args=None):
        self.inplanes = 64
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.is_att = True
        self.arch = args.backbone
        
        super(Model, self).__init__()

        ''' Backbone Net'''
        if self.arch in dir(resnet):
            self.backbone = getattr(resnet, self.arch)(pretrained=False)
        elif self.arch in dir(densenet):
            self.backbone = getattr(densenet, self.arch)(pretrained=False)
        elif self.arch in dir(senet):
            self.backbone = getattr(senet, self.arch)()
        elif self.arch == 'inception_v3':
            self.backbone = getattr(models, self.arch)(pretrained=False,aux_logits=False)
        elif self.arch in dir(models):
            self.backbone = getattr(models, self.arch)(pretrained=False)
        else:
            self.backbone = pretrainedmodels.__dict__[self.arch](num_classes=1000,pretrained=False)
            
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False
                
        if 'densenet' in self.arch:
            feat_dim = 1920
        else:
            feat_dim = 2048

        ''' Open-Domain Recognition Module '''
        self.odr_proj1 =  nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.odr_proj2 =  nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.odr_spatial =  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.odr_channel =  nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256/16, kernel_size=1, stride=1, padding=0,bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(256/16, 256, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Sigmoid(),        
        )
        self.odr_classifier = nn.Linear(int(256*(256+1)/2), num_classes)

        ''' Zero-Shot Recognition Module '''
        self.zsr_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.zsr_sem = nn.Sequential(
            nn.Linear(sf_size,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,feat_dim),
            nn.LeakyReLU()
        )
        self.zsr_aux = nn.Linear(feat_dim, num_classes)
        
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if pretrained:
            if self.arch=='resnet50':
                self.backbone.load_state_dict(torch.load('./pretrained/resnet50-19c8e357.pth'))
            elif self.arch=='resnet101':
                self.backbone.load_state_dict(torch.load('./pretrained/resnet101-5d3b4d8f.pth'))
            elif self.arch=='se_resnet152':
                self.backbone.load_state_dict(torch.load('./pretrained/se_resnet152-d17c99b7.pth'))
            elif self.arch=='senet154':
                self.backbone.load_state_dict(torch.load('./pretrained/senet154-c7b49a05.pth'))
            elif self.arch=='resnext101_32x8d':
                model_dict = self.backbone.state_dict()
                pretrained_dict = torch.load('./pretrained/resnext101_32x8d-8ba56ff5.pth')
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                self.backbone.load_state_dict(pretrained_dict)
            elif self.arch=='densenet201':
                pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
                state_dict = torch.load('./pretrained/densenet201-c1103571.pth')
                for key in list(state_dict.keys()):
                    res = pattern.match(key)
                    if res:
                        new_key = res.group(1) + res.group(2)
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
                self.backbone.load_state_dict(state_dict)
            elif self.arch=='inception_v3':
                model_dict = self.backbone.state_dict()
                pretrained_dict = torch.load('./pretrained/inception_v3_google-1a9a5a14.pth')
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                self.backbone.load_state_dict(pretrained_dict)

        if 'resne' in self.arch:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif '152' in self.arch:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        elif 'senet154' in self.arch:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        elif 'densenet' in self.arch:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.backbone.add_module('final_relu',nn.ReLU(inplace=True))
        elif 'inception_v3' in self.arch:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

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

    def forward(self, x, sf):
        ''' backbone '''
        x = self.backbone(x)
        last_conv = x 
		
        ''' ODR Module '''
        x1 = self.odr_proj1(last_conv)
        x2 = self.odr_proj1(last_conv)
        # attention
        if self.is_att:
            # att gen
            att1 = self.odr_spatial(x1)
            att2 = self.odr_channel(x2)
            # att1
            x1 = att2*x1+x1
            x1 = x1.view(x1.size(0),x1.size(1),-1)
            # att2
            x2 = att1*x2+x2
            x2 = x2.view(x2.size(0),x2.size(1),-1)
            # covariance pooling
            x1 = x1 - torch.mean(x1,dim=2,keepdim=True)
            x2 = x2 - torch.mean(x2,dim=2,keepdim=True)
            A = 1./x1.size(2)*x1.bmm(x2.transpose(1,2))
        else:
            x = x.view(x1.size(0),x2.size(1),-1)
            x = x - torch.mean(x,dim=2,keepdim=True)
            A = 1./x.size(2)*x.bmm(x.transpose(1,2))
        x = MPNCOV.SqrtmLayer(A, 5)
        x = MPNCOV.TriuvecLayer(x)
        odr_x = x.view(x.size(0), -1)
        odr_logit = self.odr_classifier(odr_x)
        
        ''' ZSR Module '''
        zsr_x = self.zsr_proj(last_conv).view(last_conv.size(0),-1)
        zsr_classifier = self.zsr_sem(sf)
        w_norm = F.normalize(zsr_classifier, p=2, dim=1)
        x_norm = F.normalize(zsr_x, p=2, dim=1)
        zsr_logit = x_norm.mm(w_norm.permute(1,0))
        zsr_logit_aux = self.zsr_aux(zsr_x)
        
        return (odr_logit,zsr_logit,zsr_logit_aux),(odr_x,zsr_x)
		
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
		
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, label, logits):
        odr_logit = logits[0]
        zsr_logit = logits[1]
        zsr_logit_aux = logits[2]
        
        ''' ODR Loss '''
        one_hot = torch.zeros_like(odr_logit)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        odr_logit = odr_logit*(1-one_hot*0.1)
        L_odr = self.cls_loss(odr_logit,label)
        
        ''' ZSL Loss '''
        idx = torch.arange(zsr_logit.size(0)).long()
        L_zsr = (1-zsr_logit[idx,label]).mean()
        
        L_aux = self.cls_loss(zsr_logit_aux,label)
        
        total_loss = L_odr + L_zsr + L_aux
		
        return total_loss,L_odr,L_zsr, L_aux
		
def d2ve(pretrained=False, loss_params=None, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(pretrained,args)
    loss_model = Loss(args)
    return model,loss_model
	
	
