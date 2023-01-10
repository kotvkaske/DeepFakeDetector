"""FTFD model."""

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class AVAM_block(nn.Module):
    """Audio-Visual Attencion Mechanism.

    Reference: https://arxiv.org/pdf/2203.05178.pdf

    Parameters
    ----------
    in_channels: int, optional (default=2)
        Input number of channels.
    out_channels: int, optional (default=1)
        Output number of channels.
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1) -> None:
        """Initialize module."""
        super().__init__()
        self.convmax = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.convavg = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.finalconv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, image: Tensor, audio: Tensor) -> Tensor:
        """Process input batch.

        image: Tensor
            Input image tensor.
        audio: Tensor
            Input audio tensor.

        Returns
        -------
        Tensor
            Attention tensor of image and audio.
        """
        F_MAX = torch.cat([
            torch.max(image, axis=1)[0].unsqueeze(dim=1),
            torch.max(audio, axis=1)[0].unsqueeze(dim=1),
        ], dim=1)
        F_AVG = torch.cat([
            torch.mean(image, axis=1).unsqueeze(dim=1),
            torch.mean(audio, axis=1).unsqueeze(dim=1),
        ], dim=1)
        F_1 = torch.cat([
            F.sigmoid(self.convavg(F_AVG)),
            F.sigmoid(self.convmax(F_MAX)),
        ], dim=1)
        M = F.sigmoid(self.finalconv(F_1))
        F_att = image * M
        return F_att


class ResBlock(nn.Module):
    """ResidualBlock.

    Reference: https://arxiv.org/pdf/1512.03385.pdf

    Parameters
    ----------
    in_channels: int, optional (default=2)
        Input number of channels.
    out_channels: int, optional (default=1)
        Output number of channels.
    downsample: bool, optional (default=False)
        If to use downsample layer.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False) -> None:
        """Initialize module."""
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Process input batch.
        
        Parameters
        ----------
        x: Tensor
            Input batch.

        Returns
        -------
        Tensor
            Processed input batch.
        """
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + shortcut
        return F.relu(x)


class MultimodalModel(nn.Module):
    """MultimodalModel.

    Reference: https://arxiv.org/pdf/2203.05178.pdf
    
    Parameters
    ----------
    in_channels (int) - количество входных каналов
    audio_size (int) - размер изображения спектрограммы
    classify (bool) - использование полносвязного слоя для классификации
    return_patch (bool) - использование патча 2х2 на выходе (PatchGan Discriminator)
    
    Forward Pass:
    Input: x (tuple: (img, audio_up, audio_d))
    Output: out (int): класс (дипфейк/нет) if classify == True else out (torch.Tensor [B,2,2]) if return_patch == True else torch.Tensor [B,256,3,3]
    
    """
    def __init__(self,in_channels,audio_size_default = 96,classify=True,return_patch = False):
        super(MultimodalModel,self).__init__()
        self.audio_size_default = audio_size_default
        self.classify = classify
        self.return_patch = return_patch
        self.avam1 = AVAM_block()
        self.avam2 = AVAM_block()
        self.avam3 = AVAM_block()
        self.avam4 = AVAM_block()
        self.avam5 = AVAM_block()
        
        self.block1 = nn.Sequential(*[AVAM_block(),ResBlock(in_channels,16,downsample=True)])
        self.block2 = nn.Sequential(*[AVAM_block(),ResBlock(16,32,downsample=True)])
        self.block3 = nn.Sequential(*[AVAM_block(),ResBlock(64,64,downsample=True)])
        self.block4 = nn.Sequential(*[AVAM_block(),ResBlock(128,128,downsample=True)])
        self.block5 = nn.Sequential(*[AVAM_block(),ResBlock(256,256,downsample=True)])                                
        
        if self.return_patch:
            self.patch_output = nn.Sequential(*[nn.Conv2d(256,1,kernel_size=2),
                                                nn.Sigmoid()])
        self.upsampled_audio1 = ResBlock(3,16,downsample=True)
        self.upsampled_audio2 = ResBlock(16,32,downsample=True)
        self.upsampled_audio3 = ResBlock(32,64,downsample=True)
        self.upsampled_audio4 = ResBlock(64,128,downsample=True)
        
        if self.audio_size_default==384:
            self.default_audio = ResBlock(3,16,downsample=True)
            self.default_audio0 = ResBlock(16,16,downsample=True)
            self.default_audio1 = ResBlock(16,32,downsample=True)
            
        elif self.audio_size_default==768:
            self.default_audio_inp = ResBlock(3,16,downsample=True)
            self.default_audio = ResBlock(16,16,downsample=True)
            self.default_audio0 = ResBlock(16,16,downsample=True)
            self.default_audio1 = ResBlock(16,32,downsample=True)
            
        self.default_audio2 = ResBlock(32,32,downsample=True)
        self.default_audio3 = ResBlock(32,64,downsample=True)
        self.default_audio4 = ResBlock(64,128,downsample=True)
        
        self.classifier = nn.Sequential(*[nn.Flatten(),nn.Linear(256*9,1000,bias=False),
                                          nn.ReLU(),nn.Dropout(p=0.5),
                                          nn.Linear(1000,100,bias=False),
                                          nn.ReLU(),nn.Dropout(p=0.5),
                                          nn.Linear(100,1,bias=False),
                                          nn.Sigmoid()])
                                          
                                          
    def forward(self,x):
 
        img,audio_up,audio_d = x
        if self.audio_size_default==384:
            audio_d = self.default_audio(audio_d)
            audio_d = self.default_audio0(audio_d)
        elif self.audio_size_default==768:
            audio_d = self.default_audio_inp(audio_d)
            audio_d = self.default_audio(audio_d)
            audio_d = self.default_audio0(audio_d)
            
        avam = self.block1(img,audio_up)
        audio_up = self.upsampled_audio1(audio_up)
        audio_d = self.default_audio1(audio_d)
        

        avam = self.block2(img,audio_up)
        audio_up = self.upsampled_audio2(audio_up)
        audio_d = self.default_audio2(audio_d)
                
        avam = torch.cat([avam,audio_d],dim=1)
        avam = self.block3(img,audio_up)
        audio_up = self.upsampled_audio3(audio_up)
        audio_d = self.default_audio3(audio_d)
        
        avam = torch.cat([avam,audio_d],dim=1)
        avam = self.block4(img,audio_up)
        audio_up = self.upsampled_audio4(audio_up)
        audio_d = self.default_audio4(audio_d)
        
        avam = torch.cat([avam,audio_d],dim=1)
        avam = self.block5(img,audio_up)
        if self.classify:
            return self.classifier(avam)
        if self.return_patch:
            return self.patch_output(avam)
        
        return avam
    
    
        