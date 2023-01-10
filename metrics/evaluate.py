import torch
import numpy as np
from torch import nn
DEVICE = torch.device('cuda')
from torch.utils.data import DataLoader


class Evaluator(nn.Module):
    """Model Evaluation
    """
    def __init__(self, type_of_model = 'av',loss_fn=nn.BCELoss(), patch_v=False):
        super(Evaluator,self).__init__()
        self.type_of_model = type_of_model
        self.patch_v = patch_v
        self.loss_fn = loss_fn
    def estimate_dataset(self,dataset,model):
        dataset = DataLoader(dataset, batch_size=20,pin_memory = True,num_workers = 8, shuffle=True)
        if self.type_of_model=='av':
            model.eval()
            val_loss,running_corrects,processed_data = 0,0,0
            for (X_batch_image, X_batch_audio_up,X_batch_audio_d), Y_batch in dataset:
                Y_batch = Y_batch.to(DEVICE)
                Y_batch = torch.tensor(Y_batch,dtype=torch.float32)
                X_batch_image = X_batch_image.to(DEVICE).squeeze(dim=1)
                X_batch_audio_up = X_batch_audio_up.to(DEVICE)
                X_batch_audio_d = X_batch_audio_d.to(DEVICE)
                with torch.set_grad_enabled(False):
                    outputs = model((X_batch_image,X_batch_audio_up,X_batch_audio_d)).squeeze(dim=1)
                    if self.patch_v:
                        preds = torch.tensor((outputs>0.5).detach().cpu(),dtype=torch.float32)
                        preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2))
                        preds = np.array(preds)
                        running_corrects += np.sum(preds == Y_batch[:,0,0].detach().cpu().numpy())
                    else:
                        preds = (outputs>0.5).detach().cpu().numpy().astype(np.float32)
                        running_corrects += np.sum(preds == Y_batch.detach().cpu().numpy())
                    processed_data += Y_batch.size(0)
                    loss = self.loss_fn(outputs,Y_batch)
                val_loss += loss / len(dataset)
            test_acc = running_corrects / processed_data
            return test_acc
        elif self.type_of_model=='v':
            running_corrects,processed_data,avg_loss = 0, 0,0
            for X_batch, Y_batch in dataset:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                with torch.set_grad_enabled(False):
                    outputs = model(X_batch).squeeze(dim=1)
                    preds = (outputs>0.5).detach().cpu().numpy().astype(np.float32)
                    if self.patch_v:
                        preds = torch.tensor((outputs>0.5).detach().cpu(),dtype=torch.float32)
                        preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2))
                        preds = np.array(preds)
                        running_corrects += np.sum(preds == Y_batch[:,0,0].detach().cpu().numpy())
                    else:
                        preds = (outputs>0.5).detach().cpu().numpy().astype(np.float32)
                        running_corrects += np.sum(preds == Y_batch.detach().cpu().numpy())
                    processed_data += Y_batch.size(0)
                    loss = self.loss_fn(outputs,Y_batch)
                avg_loss += loss / len(dataset)
            test_acc = running_corrects / processed_data
            return test_acc
            