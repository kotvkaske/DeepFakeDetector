from config import *

from datasets import *
import os
import pandas as pd
from sklearn.model_selection import train_test_split

cfg = load_config(0)


class FakeAVCeleb():
    def __init__(self, cfg):
        super(FakeAVCeleb, self).__init__()
        self.cfg = cfg
        print('ku')
        if cfg.need_preprocess:
            self.preprocess()

    def preprocess(self):
        self.split_data()
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        if 'train' not in os.listdir(cfg.save_path):
            os.mkdir(cfg.save_path+'train')
        if 'test' not in os.listdir(cfg.save_path):
            os.mkdir(cfg.save_path+'test')
        itr = 0
        for i in range(5):
            if f'sample_{itr}' not in os.listdir(cfg.save_path+'/train/'):
                os.mkdir(cfg.save_path +'/train/'+ f'sample_{itr}')
                obj = train.iloc[i]
                self.extract_video(obj['full_path']+'/'+obj['path'],
                                   cfg.save_path+'/train/'+f'sample_{itr}')

    def split_data(self):
        dataframe = pd.read_csv(cfg.dataset_path + 'meta_data.csv')
        dataframe.rename(columns={'Unnamed: 9': 'full_path'}, inplace=True)
        dataframe['full_path'] = dataframe['full_path'].apply(lambda x: '/'.join(x.split('/')[1:]))
        unique_objects = dataframe['source'].unique()
        if cfg.shuffle_data:
            np.random.shuffle(unique_objects)
        train = unique_objects[int(self.cfg.train_test_split * len(unique_objects)):]
        test = unique_objects[:int(self.cfg.train_test_split * len(unique_objects))]
        train = dataframe[dataframe['source'].isin(train)]
        test = dataframe[dataframe['source'].isin(test)]
        train.to_csv('train.csv', index=False)
        test.to_csv('test.csv', index=False)

    def extract_video(self,video_path,save_path):
        print(video_path)
        print(save_path)
        return