import torch
from torch import nn
import timm

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
        
    def forward(self,x):
        return x

class EffNetb1(nn.Module):
    def __init__(self,pretrained_folder = None,patch=False):
        super(EffNetb1,self).__init__()
        if pretrained_folder:
            self.model = torch.load(pretrained_folder)
        else:
            self.model = timm.create_model('efficientnet_b1', pretrained=True)
        if patch:
            self.model.conv_head = Identity()
            self.model.bn2 = Identity()
            self.model.act2 = Identity()
            self.model.global_pool = Identity()
            self.model.classifier = nn.Sequential(*[nn.Conv2d(320,1,kernel_size=3,stride=3),nn.Sigmoid()])
        else:
            self.model.classifier = nn.Sequential(*[nn.Linear(1280,1)],nn.Sigmoid())
    def forward(self,x):
        x = self.model(x)

        return x