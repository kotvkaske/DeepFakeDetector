import torch
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
DEVICE = torch.device('cuda')

def train_epoch(model, opt, loss_fn, data_tr, path_to_save,patch_v,DEVICE=DEVICE):
    """Train model based on Video+Audio on one epoch
        model: nn.Module
            Model to train.
        opt: torch.optim
            Model optimizer.
        loss_fn: torch.nn
            Loss function.
        data_tr: torch.utils.dataset
            Train dataset
        path_to_save: str
            Path to save model
        DEVICE: torch.device
            Device to train model
       
        Returns
        -------
        train_acc: float
            accuracy on epoch
        train_loss: flaot
            loss on train dataset
        """
    for X_batch, Y_batch in data_tr:
        model.train()
        train_loss,running_corrects,processed_data = 0,0,0
        X_batch_image,X_batch_audio_up,X_batch_audio_d = X_batch
        Y_batch = Y_batch.to(DEVICE)
        X_batch_image = X_batch_image.to(DEVICE).squeeze(dim=1)
        X_batch_audio_up,X_batch_audio_d = X_batch_audio_up.to(DEVICE),X_batch_audio_d.to(DEVICE)
        opt.zero_grad()
        outputs = model((X_batch_image,X_batch_audio_up,X_batch_audio_d)).squeeze(dim=1)
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
        train_loss += loss / len(data_tr)

    train_acc = running_corrects / processed_data
    return train_acc, train_loss
    
def eval_epoch(model,loss_fn,data_val,path_to_save,patch_v,DEVICE=DEVICE):
    """Eval model based on Video+Audio on one epoch
        model: nn.Module
            Model to train.
        loss_fn: torch.nn
            Loss function.
        data_val: torch.utils.dataset
            Test dataset
        path_to_save: str
            Path to save model
        DEVICE: torch.device
            Device to train model
        
            
        Returns
        -------
        test_acc: float
            accuracy on epoch
        test_loss: flaot
            loss on train dataset
        """
    model.eval()
    val_loss,running_corrects,processed_data = 0,0,0
    for (X_batch_image, X_batch_audio_up,X_batch_audio_d), Y_batch in data_val:
        Y_batch = Y_batch.to(DEVICE)
        Y_batch = torch.tensor(Y_batch,dtype=torch.float32)
        X_batch_image = X_batch_image.to(DEVICE).squeeze(dim=1)
        X_batch_audio_up = X_batch_audio_up.to(DEVICE)
        X_batch_audio_d = X_batch_audio_d.to(DEVICE)
        with torch.set_grad_enabled(False):
            outputs = model((X_batch_image,X_batch_audio_up,X_batch_audio_d)).squeeze(dim=1)
            
            if patch_v:
                preds = torch.tensor((outputs>0.5).detach().cpu(),dtype=torch.float32)
                preds,_ = torch.mode(torch.flatten(preds,start_dim=1,end_dim=2))
                preds = np.array(preds)
                running_corrects += np.sum(preds == Y_batch[:,0,0].detach().cpu().numpy())
                # running_corrects += np.sum(np.all(preds == Y_batch.detach().cpu().numpy(),axis=(1,2)))
            else:
                preds = (outputs>0.5).detach().cpu().numpy().astype(np.float32)
                running_corrects += np.sum(preds == Y_batch.detach().cpu().numpy())
            processed_data += Y_batch.size(0)
            loss = loss_fn(outputs,Y_batch)
        val_loss += loss / len(data_val)
    test_acc = running_corrects / processed_data
    return test_acc,val_loss

def train(model, opt, loss_fn, epochs, data_tr, data_val,batch, path_to_save,patience=10,patch_v = False,
         DEVICE = torch.device('cuda')):
    """Train+Evaluate model based on Video+Audio
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
        DEVICE: torch.device
            Device to train model
        patience: int
            early-stopping criterion value
        patch_v: bool
            If 'True', uses patch modification output
            
        Returns
        -------
        model: nn.Module
            trained and evaluated model
        """    
    data_tr = DataLoader(data_tr,
                                batch_size=batch,pin_memory = True,num_workers = 8, shuffle=True)
    data_val = DataLoader(data_val,
                                batch_size=batch,pin_memory = True,num_workers = 8, shuffle=True)
    max_loss = np.float('inf')
    max_acr = 0
    counter = 0
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:   
        for epoch in range(epochs):
            train_acc,train_loss = train_epoch(model, opt, loss_fn, data_tr, path_to_save,patch_v,
                                              DEVICE)
            print('* Epoch %d/%d' % (epoch + 1, epochs))
            pbar_outer.update(1)
            test_acc,val_loss = eval_epoch(model, loss_fn, data_val, path_to_save,patch_v,
                                          DEVICE)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,
                                           v_loss=val_loss, t_acc=train_acc, v_acc=test_acc))
            if test_acc >= max_acr :
                max_acr = test_acc
                counter = 0
                torch.save(model, path_to_save)
            else:
                print(f'сработал датчик на {epoch}-й эпохе, всего {counter+1}')
                counter += 1
                if counter == patience:
                    return model