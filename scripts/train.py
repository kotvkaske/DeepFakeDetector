import sys
import os 
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/home/kamenev_v/Deepfake_proj/DeepFakeDetection/')
import torch
import torchvision.transforms as tt
import pandas as pd
from lib.models.BaseDetector import *
from lib.models.FTFD import *
from lib.models.PFD import *
from lib.training.train_model import *
from lib.training.trainerold import *
from lib.utils import *
from lib.data.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--typeofmodel", required=True)
parser.add_argument("-p", "--patchmode", required=True)
parser.add_argument("-m", "--modelname", required=True)

args = parser.parse_args()
def train_script(model_type):
    if model_type=='v':
        model = EffNetb1(patch=args.patchmode).to(DEVICE)
        TRAIN = pd.read_csv('lib/data/FINAL_TRAIN_P2.csv')
        TEST = pd.read_csv('lib/data/FINAL_TEST_P2.csv')
        TRAIN['class'] = TRAIN['class'].apply(lambda x: torch.tensor(x,dtype=torch.float32))
        TEST['class'] = TEST['class'].apply(lambda x: torch.tensor(x,dtype=torch.float32))
        if args.patchmode:
            TRAIN['class'] = TRAIN['class'].apply(lambda x: patch_apply(x))
            TEST['class'] = TEST['class'].apply(lambda x: patch_apply(x))
        train_dataset = DeepFakeDataset(TRAIN)
        test_dataset = DeepFakeDataset(TEST)
        optimizer = torch.optim.Adam(model.parameters())
        loss = torch.nn.BCELoss()
        res = train_old(model,optimizer,loss,50,
                        train_dataset,test_dataset,args.patchmode,f'lib/models/{args.modelname}.pth')
        
    else:
        TRAIN = pd.read_csv('lib/data/FINAL_TRAIN_VA.csv')
        TEST = pd.read_csv('lib/data/FINAL_TEST_VA.csv')
        TRAIN = pd.concat([TRAIN[TRAIN['class']==1].sample(467),TRAIN[TRAIN['class']==0]])
        TEST = pd.concat([TEST[TEST['class']==1].sample(109),TEST[TEST['class']==0]])
        TRAIN['path'] = TRAIN['path'].apply(lambda x: 'lib/data/CUSTOMV2/'+x)
        TEST['path'] = TEST['path'].apply(lambda x: 'lib/data/CUSTOMV2/'+x)
        transform_image = tt.Compose([tt.Resize(96),tt.CenterCrop(96),
                                 tt.RandomHorizontalFlip(p=0.3),
                                     tt.RandomRotation(degrees=7),
                                     AddGaussianNoise(p=0.3,mean=0.5,std=0.1)])
        transform_upsampled_audio = tt.Compose([tt.Resize(96),tt.CenterCrop(96)])
        transform_default_audio = tt.Compose([tt.Resize(768),tt.CenterCrop(768)])
        classify=True
        if args.patchmode:
            classify=False
            TRAIN['class'] = TRAIN['class'].apply(lambda x: patch_apply(x))
            TEST['class'] = TEST['class'].apply(lambda x: patch_apply(x))
        model = MultimodalModel(3*3,audio_size_default=768,
                                classify=classify,return_patch=args.patchmode) ### IMAGE 
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(params = model.parameters())
        loss = nn.BCELoss()
        train_dataset = CelebVADataset(TRAIN,3,transform_image,transform_upsampled_audio,
                              transform_default_audio)
        test_dataset = CelebVADataset(TEST,3,transform_upsampled_audio,transform_upsampled_audio,
                             transform_default_audio)
        k = train(model=model,opt=optimizer,loss_fn=loss,epochs=50,
          data_tr=train_dataset,data_val=test_dataset,batch=40,patience=7,
          path_to_save=f'lib/models/{args.modelname}.pth',patch_v=args.patchmode,DEVICE=DEVICE)
        
if __name__ == '__main__':
    print(train_script(args.typeofmodel))
    
    