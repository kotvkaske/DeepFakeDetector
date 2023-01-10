from config import *

import os

cfg = load_config(0)
class FakeAVCeleb():
    def __init__(self,cfg):
        super(FakeAVCeleb,self).__init__()
        self.data_path = cfg.dataset_path

    def preprocess(self):

