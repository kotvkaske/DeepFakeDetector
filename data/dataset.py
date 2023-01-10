import os
import torch
import torchvision.io
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
from lib.utils import AddGaussianNoise

class DeepFakeDataset(Dataset):
    """
    DeepFake Dataset for building image-driven model
    Attributes
    ----------
    data : pandas.DataFrame
        Dataframe of paths to images / class
    type_of_data: bool, optional (default == 'train')
        If 'train', returns image and a label.
    """
    def __init__(self, data, type_of_data='train'):
        """Initialize module."""
        super(DeepFakeDataset, self).__init__()
        self.files_x = data['path']
        self.list_of_labels = data['class']
        self.type_of_data = type_of_data

    def __len__(self):
        """Return length of dataset"""
        return len(self.files_x)

    def __getitem__(self, index):
        """Return sample of dataset"""
        transform_train = tt.Compose([
            tt.Resize(224),
            tt.CenterCrop(224),
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            tt.RandomHorizontalFlip(p=0.3),
            tt.RandomRotation(degrees=7),
            AddGaussianNoise(p=0.5,mean=0,std=0.15)])
        transform_test = tt.Compose([
            tt.Resize(224),
            tt.CenterCrop(224),
            tt.ToTensor(),
            tt.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
        x_data = Image.open(self.files_x[index])
        if self.type_of_data == 'train':
            x_data = transform_train(x_data)
            y_data = self.list_of_labels[index]
            return x_data, y_data
        else:
            x_data = transform_test(x_data)
            return x_data

    def show_pic(self, index,type_of_data='train'):
        """Process output representation.

        index: int
            index of image.

        Returns
        x,y_data: tuple
            tuple of PIL Image and float-label
        -------
        String
           Representative form.
        """
        x = Image.open(self.files_x[index])
        y_data = self.list_of_labels[index]
        if self.type_of_data=='test':
            return x
        return x, y_data
    
    
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
    