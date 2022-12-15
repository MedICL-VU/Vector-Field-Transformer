# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:46:31 2021

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\static representation\\baselines\\VFT\\')
sys.path.insert(0, "E:\\static representation\\src\\")
sys.path.insert(0,'E:\\tools\\')

import util
import loss
import augment_arch as arch
from train_dataloader import load_single_data

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

model_root = "E:\\Model\\"
data_root = "E:\\static representation\\data\\"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_epoch = 30
enh_enc = (8,16,32,64,64)
seg_enc = (8,16,32)
nch_in = 1
nch_out = 1

model = arch.VAE(enh_enc, seg_enc, nch_in, nch_out).to(device)
#model.load_state_dict(torch.load(model_root+"LIFE_enhance_4.pt"))

DSC_loss = loss.DiceBCELoss()
CE_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

#%%
with open(data_root+"raw_data.pickle", "rb") as handle:
    im_data = pickle.load(handle)
    
im_size = [256,256]
num_sample = 100
batch_size = 4
train_list = ["octa500","rose"]

train_data = load_single_data(im_size, num_sample, im_data,
                              train_list, batch_size, reverse=False, augment=False)

#%%
softmax = nn.Softmax2d()

for epoch in range(n_epoch):
    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:
        
        for step, (x,y) in enumerate(train_data):    
            
            model.train()
            
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            latent, pred_y = model(x)
            
            celoss = CE_loss(pred_y,y)
            dscloss = DSC_loss(torch.argmax(softmax(pred_y),dim=1),y)
            losses = celoss + dscloss
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred_y_np = softmax(pred_y).detach().cpu().numpy()
                opt = np.argmax(pred_y_np,axis=1)
                gt = y.detach().cpu().numpy()
                Dice = util.dice(opt[0,:,:],gt[0,:,:])
                
                pbar.update(1)
                pbar.set_description('Epoch: %d. CE-Loss: %.4f. DSC-loss: %.4f. Dice: %.4f.'
                                     % (epoch+1, celoss.item(), dscloss.item(), Dice))
                
                if step % (len(train_data)-1) == 0 and step != 0:
                    im_x = x[0,0,:,:].detach().cpu().numpy()
                    im_latent = latent[0,0,:,:].detach().cpu().numpy()
                    im_y = gt[0,:,:]
                    im_pred = opt[0,:,:]
                                        
                    fig, ax = plt.subplots(2,2,figsize=(8,8))
                    ax[0,0].imshow(im_x, cmap="gray"),ax[0,0].axis("off")
                    ax[0,1].imshow(im_y, cmap="gray"),ax[0,1].axis("off")
                    ax[1,0].imshow(im_latent, cmap="gray"),ax[1,0].axis("off")
                    ax[1,1].imshow(im_pred, cmap="gray"),ax[1,1].axis("off")
                    
                    fig.suptitle("train visual")
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.95)
                    plt.show()
                
    scheduler.step()

name = 'VFT_augment(octa)_4.pt'
torch.save(model.state_dict(),model_root+name)
