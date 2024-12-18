
import copy
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
from tqdm import tqdm
from loss import DiceLoss
from torch.utils.data import DataLoader, random_split
import loadData

def train(
        model, 
        device, 
        dataset,
        num_classes,
        base_lr=0.01,
        batch_size: int = 1,
        epochs=100):
    #summary(model, input_size=(4,256,256))
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    val_percent = 0.33
    val_size = int(len(dataset)*val_percent)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(0)) # 该生成器设置的种子使每次运行该文件都会产生同样的分组
    
    
    
    
    
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, drop_last=False, num_workers=1
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, drop_last=False, num_workers=1
    )
    # train_epochs_loss = 0 # average loss of each epoch
    step = 0
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    dsc_loss = DiceLoss(num_classes) # didn't work
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    iter_num = 0
    max_iterations = epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    for epoch in range(1, epochs+1):
    #for epoch in tqdm(range(epochs), total=epochs):
        train_loss = 0 # loss of every data in each epoch 
        model.train()
        with tqdm(total=train_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                x, y_true = batch
                
                testyt = y_true>=14
                #print("test", y_true.size(), y_true[testyt])
                y_true = y_true.squeeze()
                y_true = y_true.unsqueeze(0)
                y_true = y_true.long()
                
                #print("x.shape = ", x.shape)
                #print("x.min(), x.max() = ", x.min(), x.max())
                #print("y.shape = ", y_true.shape)
                #print("torch.unique(y) = ", torch.unique(y_true))
                x, y_true = x.to(device=device), y_true.to(device=device)
                y_pred = model(x)
                optimizer.zero_grad()
                ce_loss = criterion(y_pred, y_true)
                #assert False, print(str(y_pred))
                #print("min: {0}, max: {1}".format(y_true.long().min(), y_true.long().max()))
                #dice_loss = dsc_loss(y_pred, y_true, softmax=True)
                #loss = 0.5*ce_loss+0.5*dice_loss
                loss = ce_loss
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()
                pbar.update(1)
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                #lr_ = base_lr * 0.9**iter_num
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
        train_epoch_loss = train_loss/len(train_loader)
        print("\n epoch: {}, train_loss: {:.4f}".format(epoch, train_epoch_loss))

        #=============valid==============
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y_true = batch
                
                y_true = y_true.squeeze()
                y_true = y_true.unsqueeze(0)
                y_true = y_true.long()
                
                x, y_true = x.to(device=device), y_true.to(device=device)
                y_pred = model(x)
                ce_loss = criterion(y_pred, y_true)
                #dice_loss = dsc_loss(y_pred, y_true, softmax=True)
                #loss = 0.5*ce_loss+0.5*dice_loss
                loss = ce_loss
                valid_loss+=loss.item()
        valid_epoch_loss = valid_loss/len(val_loader)
        print("\n epoch: {}, val_loss: {:.4f}".format(epoch, valid_epoch_loss))
        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model
