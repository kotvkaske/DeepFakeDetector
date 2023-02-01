from data_preprocess import *
from data.dataset import CelebVADataset
from models.FTFD import *
from training.train_model import *

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss

cfg = load_config(0)

PreprocessingStep = FakeAVCelebPreprocessing(cfg)
''' Data Loading'''
train_data = CelebVADataset(cfg, 'train')
test_data = CelebVADataset(cfg, 'test')

''' Model Loading '''

model = MultimodalModel(9)
optimizer = Adam(params = model.parameters())
loss =

''' Train Process'''

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
