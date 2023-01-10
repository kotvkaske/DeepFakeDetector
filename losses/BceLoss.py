import torch

def bce_loss(y_real, y_pred):
    res = y_pred - y_pred*y_real + torch.log(1+torch.exp(-y_pred))  
    return torch.mean(res)