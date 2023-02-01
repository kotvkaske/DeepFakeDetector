from data_preprocess import *
from data.dataset import CelebVADataset

cfg = load_config(0)

PreprocessingStep = FakeAVCelebPreprocessing(cfg)
data = CelebVADataset(cfg,'train')
print(data[0])

