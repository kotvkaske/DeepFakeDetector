from time import time
import torch
import numpy as np
from torch.utils.data import DataLoader
DEVICE = torch.device('cuda')


def train_old(model, opt, loss_fn, epochs, data_tr, data_val,patch_v, path_to_save,patience=3):
    """Train+Evaluate model based on Video without sound / Image
        model: nn.Module
            Model to train.
        opt: torch.optim
            Model optimizer.
        loss_fn: torch.nn
            Loss function.
        data_tr: torch.utils.dataset
            Train dataset
        data_val: torch.utils.dataset
            Test dataset
        path_to_save: str
            Path to save model
        patch_v: bool
            If 'True', uses patch modification output
        patience: int, optional (default = 3)
            early-stopping criterion value
            
        Returns
        -------
        model: nn.Module
            trained and evaluated model
        """   
    data_tr = DataLoader(data_tr, batch_size=20,pin_memory = True,num_workers = 8, shuffle=True)
    data_val = DataLoader(data_val, batch_size=20,pin_memory = True,num_workers = 8, shuffle=True)
    max_loss = np.float('inf')
    max_acr = 0
    counter = 0
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch + 1, epochs))
        avg_loss = 0
        patience = 3
        model.train()
        running_corrects = 0
        processed_data = 0
        for X_batch, Y_batch in data_tr:
            Y_batch = Y_batch.to(DEVICE)
            X_batch = X_batch.to(DEVICE)
            opt.zero_grad()
            outputs = model(X_batch).squeeze(dim=1)
            loss = loss_fn(outputs,Y_batch)
            loss.backward()
            opt.step()
        if patch_v:
            preds = torch.tensor((outputs>0.5).detach().cpu(),dtype=torch.float32)
            preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2))
            preds = np.array(preds)
            running_corrects += np.sum(preds == Y_batch[:,0,0].detach().cpu().numpy())
        else:
            preds = (outputs>0.5).detach().cpu().numpy().astype(np.float32)
            running_corrects += np.sum(preds == Y_batch.detach().cpu().numpy())         
        processed_data += Y_batch.size(0)
        avg_loss += loss / len(data_tr)
        
        train_acc = running_corrects / processed_data
        print('train_acc: %f' % train_acc)
        toc = time()
        print('train_loss: %f' % avg_loss)
        avg_loss = 0
        model.eval()
        
        running_corrects = 0
        processed_data = 0
        for X_batch, Y_batch in data_val:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            with torch.set_grad_enabled(False):
                outputs = model(X_batch).squeeze(dim=1)
                preds = (outputs>0.5).detach().cpu().numpy().astype(np.float32)
                if patch_v:
                    preds = torch.tensor((outputs>0.5).detach().cpu(),dtype=torch.float32)
                    preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2))
                    preds = np.array(preds)
                    running_corrects += np.sum(preds == Y_batch[:,0,0].detach().cpu().numpy())
                else:
                    preds = (outputs>0.5).detach().cpu().numpy().astype(np.float32)
                    running_corrects += np.sum(preds == Y_batch.detach().cpu().numpy())
                processed_data += Y_batch.size(0)
                loss = loss_fn(outputs,Y_batch)
            avg_loss += loss / len(data_val)
        test_acc = running_corrects / processed_data
        print('test_acc: %f' % test_acc)
        print('val_loss: %f' % avg_loss)
        if test_acc >= max_acr :
            max_acr = test_acc
            counter = 0
            torch.save(model, path_to_save)


        else:
            print(f'сработал датчик на {epoch}-й эпохе, всего {counter+1}')
            counter += 1
            if counter == patience:
                return model

            