from config import *
import torchvision.transforms as tt


class Transformation():
    def __init__(self, mode, cfg):
        self.cfg = cfg
        assert (mode in ['train', 'test']), "Unknown Mode"
        if mode == 'train':
            self.transformation_image = tt.Compose(
                [tt.ToPILImage(), tt.Resize(cfg.image_size), tt.CenterCrop(cfg.image_size),
                 tt.ToTensor(),
                 tt.RandomHorizontalFlip(p=0.3),
                 tt.RandomRotation(degrees=7)])
            self.transformation_spc_concat = tt.Compose(
                [tt.Resize(cfg.spectrgram_size), tt.CenterCrop(cfg.spectrgram_size)])
            self.transformation_spc_default = tt.Compose(
                [tt.Resize(cfg.upsampled_spectgram_size), tt.CenterCrop(cfg.upsampled_spectgram_size)])

    def transform(self, x, type_of_img):
        return

    # transform_image = tt.Compose([tt.ToPILImage(),tt.Resize(image_size),tt.CenterCrop(image_size),


#                               tt.ToTensor(),
#                                  tt.RandomHorizontalFlip(p=0.3),
#                                      tt.RandomRotation(degrees=7),
#                                      AddGaussianNoise(p=0.5,mean=0,std=0.1)])
transform_image = tt.Compose([tt.Resize(image_size), tt.CenterCrop(image_size)])
# tt.RandomHorizontalFlip(p=0.3),
#     tt.RandomRotation(degrees=7),
#     AddGaussianNoise(p=0.3,mean=0.5,std=0.1)])
transform_upsampled_audio = tt.Compose([tt.Resize(audio_size_up), tt.CenterCrop(audio_size_up)])
transform_default_audio = tt.Compose([tt.Resize(audio_size_d), tt.CenterCrop(audio_size_d)])
