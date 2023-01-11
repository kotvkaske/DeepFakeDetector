from data_preprocess import *

cfg = load_config(0)
dataset = FakeAVCeleb(cfg)

batch_size = 20
image_size = 224

