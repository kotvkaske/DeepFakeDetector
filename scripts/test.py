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
from lib.metrics.evaluate import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--typeofmodel", required=True)
parser.add_argument("-p", "--patchmode", required=True)
parser.add_argument("-m", "--modelname", required=True)

args = parser.parse_args()
def test_script(model_type):
    if model_type=='v':
        model = torch.load(f'lib/models/{args.modelname}.pth').to(DEVICE)
        TEST = pd.read_csv('lib/data/FINAL_TEST_P2.csv')
        TEST['class'] = TEST['class'].apply(lambda x: torch.tensor(x,dtype=torch.float32))
        if args.patchmode:
            TEST['class'] = TEST['class'].apply(lambda x: patch_apply(x))
        test_dataset = DeepFakeDataset(TEST)
        loss = torch.nn.BCELoss()

        checker = Evaluator(type_of_model='model_type',
                            loss_fn=nn.BCELoss(),patch_v=args.patchmode)
        checker.estimate_dataset(test_dataset,model)
        
    else:
        TEST = pd.read_csv('lib/data/FINAL_TEST_VA.csv')
        TEST = pd.concat([TEST[TEST['class']==1].sample(109),TEST[TEST['class']==0]])
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
            TEST['class'] = TEST['class'].apply(lambda x: patch_apply(x))
        model = torch.load(f'lib/models/{args.modelname}.pth').to(DEVICE)
        train_dataset = CelebVADataset(TRAIN,3,transform_image,transform_upsampled_audio,
                              transform_default_audio)
        test_dataset = CelebVADataset(TEST,3,transform_upsampled_audio,
                                      transform_upsampled_audio,
                             transform_default_audio)
        checker = Evaluator(type_of_model='model_type',
                            loss_fn=nn.BCELoss(),patch_v=args.patchmode)
        checker.estimate_dataset(test_dataset,model)
        
if __name__ == '__main__':
    print(test_script(args.typeofmodel))
    
    