import torch
import torchvision.utils
from PIL import Image as IMG

from config import *
from utils import *

from facenet_pytorch import MTCNN
from datasets import *
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.io import read_video

cfg = load_config(0)


class FakeAVCelebPreprocessing():
    def __init__(self, cfg):
        super(FakeAVCelebPreprocessing, self).__init__()
        self.cfg = cfg
        if cfg.need_preprocess:
            self.preprocess()

    def preprocess(self):
        self.split_data_and_save()
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        if 'train' not in os.listdir(cfg.save_path):
            os.mkdir(cfg.save_path + 'train')
        if 'test' not in os.listdir(cfg.save_path):
            os.mkdir(cfg.save_path + 'test')
        if 'fake' not in os.listdir(cfg.save_path + 'train/'):
            os.mkdir(cfg.save_path + 'train/fake')
        if 'real' not in os.listdir(cfg.save_path + 'train/'):
            os.mkdir(cfg.save_path + 'train/real')
        if 'fake' not in os.listdir(cfg.save_path + 'test/'):
            os.mkdir(cfg.save_path + 'test/fake')
        if 'real' not in os.listdir(cfg.save_path + 'test/'):
            os.mkdir(cfg.save_path + 'test/real')
        itr = 0
        for i in range(int(cfg.dataset_size * cfg.train_test_split)):
            try:
                obj = train.iloc[i]
                if obj.method != 'real':
                    os.mkdir(cfg.save_path + 'train/fake/' + f'sample_{itr}')
                    self.extract_video(cfg.dataset_path + obj['full_path'] + '/' + obj['path'],
                                       cfg.save_path + 'train/fake/' + f'sample_{itr}')
                else:
                    os.mkdir(cfg.save_path + 'train/real/' + f'sample_{itr}')
                    self.extract_video(cfg.dataset_path + obj['full_path'] + '/' + obj['path'],
                                       cfg.save_path + 'train/real/' + f'sample_{itr}')

                itr += 1
            except:
                continue
        itr = 0
        for i in range(int(cfg.dataset_size * (1 - cfg.train_test_split))):
            try:
                obj = test.iloc[i]
                if obj.method != 'real':
                    os.mkdir(cfg.save_path + 'test/fake/' + f'sample_{itr}')
                    self.extract_video(cfg.dataset_path + obj['full_path'] + '/' + obj['path'],
                                       cfg.save_path + 'test/fake/' + f'sample_{itr}')
                else:
                    os.mkdir(cfg.save_path + 'test/real/' + f'sample_{itr}')
                    self.extract_video(cfg.dataset_path + obj['full_path'] + '/' + obj['path'],
                                       cfg.save_path + 'test/real/' + f'sample_{itr}')
                itr += 1
            except:
                continue

    def split_data_and_save(self):
        dataframe = pd.read_csv(cfg.dataset_path + 'meta_data.csv')
        dataframe.rename(columns={'Unnamed: 9': 'full_path'}, inplace=True)
        dataframe['full_path'] = dataframe['full_path'].apply(lambda x: '/'.join(x.split('/')[1:]))
        unique_objects = dataframe['source'].unique()
        if cfg.shuffle_data:
            np.random.shuffle(unique_objects)

        train = unique_objects[:int(self.cfg.train_test_split * len(unique_objects))]
        test = unique_objects[int(self.cfg.train_test_split * len(unique_objects)):]
        train = dataframe[dataframe['source'].isin(train)]
        test = dataframe[dataframe['source'].isin(test)]
        train.to_csv('train.csv', index=False)
        test.to_csv('test.csv', index=False)

    def extract_video(self, video_path, save_path):
        frames, audio, info = read_video(video_path, pts_unit='sec', start_pts=1, end_pts=1.5)
        if audio.size(0) != 1:
            audio = audio[0].unsqueeze(dim=0)
        graph_spectrogram(audio.squeeze(), info['audio_fps'], save_path + "/spec.png")
        res = self.predict_video(frames)
        itr_save = 0

        for i in range(len(res)):
            im = IMG.fromarray(np.transpose(res[i], (1, 2, 0)))
            im.save(save_path + f"/img_{itr_save}.png")
            itr_save += 1
        return

    def predict_video(self, frames):
        """Process Image/Video classification
        video: mkv/avi/...
            Input video.

        Returns
        -------
        Whether video fake or not
        """
        res, confidence = MTCNN()(frames, return_prob=True)
        confidence = torch.from_numpy(confidence)
        res = torch.stack(res).numpy()
        res = ((res - np.min(res)) / (np.max(res) - np.min(res)) * 255).astype(np.uint8)
        res = torch.tensor(res)
        indices = (confidence > cfg.face_confidence).squeeze(dim=1)
        faces = res[indices].numpy()
        return faces[:cfg.frames_per_video]
