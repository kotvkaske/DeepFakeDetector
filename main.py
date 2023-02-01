from data_preprocess import *
from data.dataset import CelebVADataset
from models.FTFD import *
from training.train_model import *

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss

DEVICE = torch.device('cuda')
cfg = load_config(0)

PreprocessingStep = FakeAVCelebPreprocessing(cfg)
''' Data Loading'''
train_data = CelebVADataset(cfg, 'train')
test_data = CelebVADataset(cfg, 'test')

''' Model Loading '''

model = MultimodalModel(9).to(DEVICE)
optimizer = Adam(params=model.parameters())
loss = nn.BCELoss()

''' Train Process'''

train(model, optimizer, loss, 5, train_data, test_data, 8, 'model.pth', 2)
