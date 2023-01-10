import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np



class AddGaussianNoise:
    def __init__(self,p=0.5, mean=0., std=1.):
        self.std = std
        self.mean = mean
        self.p = torch.tensor(p,dtype=torch.float32)
    def __call__(self, tensor):
        probability = torch.bernoulli(input=self.p)
        if probability==torch.tensor(1.):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
def getImagePaths(path):
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names

def patch_apply(x):
    return torch.tensor([[x,x],[x,x]],dtype=torch.float32)



def graph_spectrogram(wave,rate,path):
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=wave, Fs=rate, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig(path, dpi=90, frameon='false')
    return 1

def patch_apply(x):
    return torch.tensor([[x,x],[x,x]],dtype=torch.float32)