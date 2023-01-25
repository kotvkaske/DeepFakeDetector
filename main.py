from data_preprocess import *
from data.dataset import CelebVADataset

cfg = load_config(0)
FakeAVCelebDataset = FakeAVCeleb(cfg, mode='train')
