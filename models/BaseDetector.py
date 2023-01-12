import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision.io import read_video
from torchvision.io import read_image
import torchvision
from facenet_pytorch import MTCNN
from utils import *
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda')

class DeepFakeDetector(nn.Module):
    """Multimodal detector for deep fake detection

    Attributes
    ----------
    num_of_input_imgs: int, optional (default=3)
        Number of input images to classifier.
    confidence_face: float, optional(default=0.95)
        threshold to use face extraction.
    patch_mode: boolean, optional (default=False)
        if 'True', uses patch modification.
    device: bool, optional (default = torch.device('cuda')
        whether use GPU or not.
    """
    def __init__(self,num_of_input_imgs = 3,confidence_face = 0.95,patch_mode = False,DEVICE=DEVICE):
        """Initialize module."""
        super(DeepFakeDetector,self).__init__()
        self.extractor = MTCNN()
        self.patch_mode = patch_mode
        if self.patch_mode:
            self.classifier1 = torch.load('lib/models/b1_patch.pth')
            self.classifier2 = torch.load('lib/models/FTFD_768_final_patch.pth')
        else:
            self.classifier1 = torch.load('lib/models/b1_default.pth')
            self.classifier2 = torch.load('lib/models/FTFD_768_final.pth')
        self.num_of_input_imgs = num_of_input_imgs
        self.confidence_face = confidence_face
        self.DEVICE = DEVICE
        
    
    def apply_answer(self,x):
        """Process output representation.

        x: Tensor
            Input image tensor.

        Returns
        -------
        String
           Representative form.
        """
        if x>0.5:
            return f'This is fake'
        else:
            return f'This is real'
        
    def image_classifier(self,x):
        """Process Image classification
        x: Tensor
            Input image tensor.
            
        Returns
        -------
        Whether image fake or not
        """
        x = x.to(DEVICE)
        x = tt.Compose([tt.Resize(224),tt.CenterCrop(224),tt.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])(x)
        result = self.classifier1(x.unsqueeze(dim=0))
        if self.patch_mode:
            result = torch.flatten(result,start_dim=1,end_dim=2)
            preds = torch.tensor((result>0.5).detach().cpu(),dtype=torch.float32)
            preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2)) 
            return self.apply_answer(preds.numpy())
        else:
            return self.apply_answer(result)
    
    def av_classifier(self,faces,audio,info):
        """Process Video classification
        faces: Tensor
            Input image tensor of faces.
        audio: Tensor
            Input Audio spectrogram image.
        info: Dict
            Input Video and Audio information.
            
        Returns
        -------
        Whether video fake or not
        """
        transform_image = tt.Compose([tt.Resize(96),tt.CenterCrop(96)])
        transform_audio = tt.Compose([tt.Resize(768),tt.CenterCrop(768)])
        faces = faces[:self.num_of_input_imgs]
        faces = [transform_image(i) for i in faces]
        faces = torch.stack((faces))
        faces = torch.tensor(faces.view(self.num_of_input_imgs*3,96,96),dtype=torch.float32).to(DEVICE)
        if audio.size(0)!=1:
            audio = audio[0].unsqueeze(dim=0)
        graph_spectrogram(audio.squeeze(),info['audio_fps'],'spectrogram.jpg')
        spectrgram = torch.tensor(torchvision.io.read_image('spectrogram.jpg')/255,dtype = torch.float32).to(DEVICE)
        os.remove('spectrogram.jpg')

        result = self.classifier2((faces.unsqueeze(dim=0),transform_image(spectrgram).unsqueeze(dim=0),
                                  transform_audio(spectrgram).unsqueeze(dim=0)))
        if self.patch_mode:
            result = torch.flatten(result,start_dim=1,end_dim=2)
            preds = torch.tensor((result>0.5).detach().cpu(),dtype=torch.float32)
            preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2))
        return self.apply_answer(preds.detach().cpu().numpy())
    

    
    def predict_image(self,img):
        img = (read_image(img).to(DEVICE))/255
        
        x = tt.Compose([tt.Resize(224),tt.CenterCrop(224),
                        tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img)
        result = self.classifier1(x.unsqueeze(dim=0))
        if self.patch_mode:
            result = torch.flatten(result,start_dim=1,end_dim=2)
            
            preds = torch.tensor((result>0.5).detach().cpu(),dtype=torch.float32)
            preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2)) 
            
            return self.apply_answer(preds.numpy())
        else:
            return self.apply_answer(result)
        
        
    
    

        
        

