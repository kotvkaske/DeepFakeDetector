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
        else:
            self.transformation_image = tt.Compose(
                [tt.ToPILImage(), tt.Resize(cfg.image_size), tt.CenterCrop(cfg.image_size),
                 tt.ToTensor()])

        self.transformation_spc_concat = tt.Compose(
            [tt.Resize(cfg.spectrgram_size), tt.CenterCrop(cfg.spectrgram_size)])
        self.transformation_spc_default = tt.Compose(
            [tt.Resize(cfg.upsampled_spectgram_size), tt.CenterCrop(cfg.upsampled_spectgram_size)])

    def transform_image(self, x):
        return self.transformation_image(x)

    def transform_spc_concat(self, x):
        return self.transformation_spc_concat(x)

    def transform_spc_default(self, x):
        return self.transformation_spc_default(x)
