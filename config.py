import numpy as np
import torch

class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id):
    cfg = Config()
    ''' Experiment '''
    cfg.experiment_idx = exp_id

    cfg.name = 'DF'

    ''' 
    **************************************** Paths ****************************************
    save_path: results will be saved at this location
    dataset_path: dataset must be stored here.
    '''
    cfg.save_path = '../../Загрузки/FakeAVCeleb_v1.2/CUSTOM/'  # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    cfg.dataset_path = '../../Загрузки/FakeAVCeleb_v1.2/' # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    cfg.need_preprocess = True
    assert cfg.save_path != None, "Set cfg.save_path in config.py"
    assert cfg.dataset_path != None, "Set cfg.dataset_path in config.py"

    ''' 
    ************************************************************************************************
    '''

    ''' Dataset '''
    # input should be cubic. Otherwise, input should be padded accordingly.
    cfg.train_test_split = 0.7
    cfg.frames_per_video = 3
    cfg.shuffle_data = True
    ''' Model '''
    cfg.face_confidence = 0.9
    cfg.first_layer_channels = 16
    cfg.num_input_channels = 1
    cfg.steps = 4

    # Only supports batch size 1 at the moment.
    cfg.batch_size = 1

    cfg.num_classes = 2
    cfg.batch_norm = True
    cfg.graph_conv_layer_count = 4

    ''' Optimizer '''
    cfg.learning_rate = 1e-4

    ''' Training '''
    cfg.numb_of_itrs = 300000
    cfg.eval_every = 1000  # saves results to disk



    return cfg