import os
import torch
import torchvision.io
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
# from lib.utils import AddGaussianNoise

class CelebVADataset(Dataset):
    """
    CelebVA Dataset 
    Attributes
    ----------
    data : pandas.DataFrame
        Dataframe of paths to images / class.
    T : int
        Input channel of concatenated pictures.
    transform_image : torch.transforms
        Image preprocessing.
    transform_upsampled_audio : torch.transforms
        Upsampled audio preprocessing.
    transform_default_audio : torch.transforms
        Default audio preprocessing.
    img_only : bool
        If True, returns only one image and a label.
    """
    def __init__(self,data,T,transform_image,transform_upsampled_audio,
                 transform_default_audio,img_only = False):
        """Initialize module."""
        super(CelebVADataset, self).__init__()
        self.data = data
        self.T = T
        self.transform_image, self.transform_upsampled_audio = transform_image, transform_upsampled_audio
        self.transform_default_audio = transform_default_audio
        self.img_only = img_only
        
    def __len__(self):
        """Return length of dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """Return sample of dataset"""
        sample = self.data.iloc[index]
        class_ind = torch.tensor(sample['class'],dtype=torch.float32)
        sample_path = sample['path']
        l = os.listdir(sample_path)
        l = np.array(l)
        spec = l[np.array([i.split('/')[-1].startswith('spec') for i in l])].item()
        path_to_images = l[np.array([i.split('/')[-1].startswith('img') for i in l])][:self.T]
        if self.transform_image:
            images = [self.transform_image(torchvision.io.read_image(sample_path+'/'+i)) for i in path_to_images]
        else:
            images = [torchvision.io.read_image(sample_path+'/'+i) for i in path_to_images]
        if self.transform_upsampled_audio:
            spec_up = self.transform_upsampled_audio(torchvision.io.read_image(sample_path+'/'+spec))
        else:
            spec_up = torchvision.io.read_image(sample_path+'/'+spec)
            
        if self.transform_default_audio:
            spec_d = self.transform_default_audio(torchvision.io.read_image(sample_path+'/'+spec))
        else:
            spec_d = torchvision.io.read_image(sample_path+'/'+spec)
        if self.img_only:
            return images[0]/255, class_ind
        images = torch.stack((images))
        images = images.view(self.T*3,96,96)
        images = images/255
        spec_up = spec_up/255
        spec_d = spec_d/255
        return (images, spec_up,spec_d), class_ind
    