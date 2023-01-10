import torch
from BceLoss import bce_loss

def focal_loss(y_pred, y_real, alpha=0.8, gamma=2):
    y_pred = y_pred.view(-1)
    y_real = y_real.view(-1)
    BCE = bce_loss(y_pred, y_real)
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE

