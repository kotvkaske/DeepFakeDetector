import os

import pandas as pd
import torch
import torchvision.io
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt

from transform import *


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

    def __init__(self, cfg, mode):
        """Initialize module."""
        super(CelebVADataset, self).__init__()
        self.mode = mode
        self.cfg = cfg
        assert (mode in ['train', 'test']), "Unknown Data Mode"
        self.data = self.cfg.save_path + mode + '/'
        true_samples = os.listdir(self.data + 'real/')
        true_samples = [self.data + 'real/' + i for i in true_samples]
        fake_samples = os.listdir(self.data + 'fake/')
        fake_samples = [self.data + 'fake/' + i for i in fake_samples]
        true_samples = pd.DataFrame(np.array([true_samples,
                                              [1 for i in range(len(true_samples))]]).T,
                                    columns=['path', 'is_real'])
        fake_samples = pd.DataFrame(np.array([fake_samples,
                                              [0 for i in range(len(fake_samples))]]).T,
                                    columns=['path', 'is_real'])
        self.pd_data = pd.concat([true_samples, fake_samples], ignore_index=True)
        self.pd_data = self.pd_data.sample(frac=1)

    def __len__(self):
        """Return length of dataset"""
        return len(self.pd_data)

    def __getitem__(self, index):
        """Return sample of dataset"""
        sample = self.pd_data.iloc[index]
        class_ind = torch.tensor(int(sample['is_real']), dtype=torch.float32)
        sample_path = sample['path']
        l = os.listdir(sample_path)
        l = np.array(l)

        spec = l[np.array([i.split('/')[-1].startswith('spec') for i in l])].item()
        path_to_images = l[np.array([i.split('/')[-1].startswith('img') for i in l])]
        transf = Transformation(self.mode, self.cfg)
        images = torch.stack(
            [transf.transformation_image(torchvision.io.read_image(sample_path + '/' + i)) for i in path_to_images])
        spec_up = transf.transformation_spc_concat(torchvision.io.read_image(sample_path + '/' + spec))
        spec_d = transf.transformation_spc_default(torchvision.io.read_image(sample_path + '/' + spec))

        images = images.view(self.T * 3, 96, 96)
        images = images / 255
        spec_up = spec_up / 255
        spec_d = spec_d / 255
        return (images, spec_up, spec_d), class_ind
