from data_preprocess import *
from data.dataset import CelebVADataset

from torch.utils.data import DataLoader

cfg = load_config(0)

PreprocessingStep = FakeAVCelebPreprocessing(cfg)

train_data = CelebVADataset(cfg, 'train')
test_data = CelebVADataset(cfg, 'test')

''' Train Process'''

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

