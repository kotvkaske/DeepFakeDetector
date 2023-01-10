import sys
import os 
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/home/kamenev_v/Deepfake_proj/DeepFakeDetection/')
from lib.models.BaseDetector import *
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pathfile", required=True)
args = parser.parse_args()
def inference(path_to_file):
    mod = DeepFakeDetector(num_of_input_imgs=3,
                       confidence_face=0.95,patch_mode=True)
    extension = path_to_file.split('.')[-1]
    if (extension=='mkv') or (extension=='avi') or (extension=='mp4'):
        return mod.predict_video(path_to_file)
    elif (extension=='jpg') or (extension=='jpeg') or (extension=='png'):
        return mod.predict_image(path_to_file)
    
    return 'Non-existing extension'
if __name__ == '__main__':
    print(inference(args.pathfile))
    
    