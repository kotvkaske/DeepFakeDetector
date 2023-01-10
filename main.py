import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda')

from data.dataset import *
from models.PFD import *
from utils import *
from training.train_model import *
from models.FTFD import *
from models.BaseDetector import *
from training.trainerold import train_old

batch_size = 20
image_size = 224



TRAIN = pd.read_csv('data/FINAL_TRAIN_P2.csv')
TEST = pd.read_csv('data/FINAL_TEST_P2.csv')
# TRAIN['path']=TRAIN['path'].apply(lambda x: 'lib/data/MixedData/'+x)
# TEST['path']=TEST['path'].apply(lambda x: 'lib/data/MixedData/'+x)
TRAIN['class'] = TRAIN['class'].apply(lambda x: torch.tensor(x,dtype=torch.float32))
TEST['class'] = TEST['class'].apply(lambda x: torch.tensor(x,dtype=torch.float32))

TRAIN['class'] = TRAIN['class'].apply(lambda x: patch_apply(x))
TEST['class'] = TEST['class'].apply(lambda x: patch_apply(x))
train_dataset = DeepFakeDataset(TRAIN)
test_dataset = DeepFakeDataset(TEST)

model = EffNetb1(patch=False,pretrained_folder = 'models/b1_patch.pth')
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss = torch.nn.BCELoss()
res = train_old(model,optimizer,loss,50,train_dataset,test_dataset,False,'models/b1_default.pth')

from metrics.evaluate import Evaluator
checker = Evaluator(type_of_model='av',loss_fn=nn.BCELoss(),patch_v=True)
checker.estimate_dataset(test_dataset,model)
checker