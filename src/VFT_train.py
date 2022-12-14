# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 05:08:38 2022

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\static representation\\baselines\\VFT\\')
sys.path.insert(0, "E:\\static representation\\src\\")
sys.path.insert(0,'E:\\tools\\')

import util
import loss
import models
import VFT_arch as net
from train_dataloader import load_single_data, load_vf_data

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

data_root = "E:\\static representation\\data\\"
model_root = "E:\\model\\"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = net.VFTransformer_v3([2,16,32,64,64]).to(device)
#model = models.res_UNet([8,16,16,32,128], 2, 2).to(device)
#model.load_state_dict(torch.load(model_root + "VFTv2_train_on_fundus_2021-11-09.pt"))

#num_p1 = count_parameters(model1)
#num_p2 = count_parameters(model2)

DSC_loss = loss.DiceBCELoss()
CE_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

#%% load data

with open(data_root + "vf_wo_optimize.pickle", "rb") as handle:
    vf_data = pickle.load(handle)

#name_list = ["augments(octa)_0","augments(octa)_1","augments(octa)_2",
#             "augments(octa)_3","augments(octa)_4","augments(octa)_5"]
#name_list = ["augments_0"]

n_epoch = 100
im_size = [256,256]
num_sample = 30
batch_size = 6

#train_loader = load_single_data(im_size, num_sample, vf_data, 
#                              name_list, batch_size, reverse=True)
train_loader = load_vf_data(im_size, num_sample, vf_data, batch_size)

#%%
class SampleMatrix(nn.Module):
    def __init__(self,):
        super(SampleMatrix, self).__init__()
        self.softmax = nn.Softmax2d()
        
    def forward(self, tensor, t_type, idx=0):
        if t_type == "pred":
            pred_tensor = torch.argmax(self.softmax(tensor),dim=1)
            matrix = pred_tensor[idx,:,:].detach().cpu().numpy()
        elif t_type == "latent":
            matrix = tensor[idx,0,:,:].detach().cpu().numpy()
        elif t_type == "gt":
            matrix = tensor[idx,:,:].detach().cpu().numpy()
        else:
            raise ValueError
        
        return matrix

#%% train
softmax = nn.Softmax2d()
get_sample = SampleMatrix()

for epoch in range(n_epoch):
    values = range(len(train_loader))
    with tqdm(total=len(values)) as pbar:
        
        for step, (im,x,y) in enumerate(train_loader):
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            
            model.train()
            pred_y, latent = model(x)
            pred = torch.argmax(softmax(pred_y), dim=1)
            
            losses = CE_loss(pred_y, y) + DSC_loss(pred, y)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            with torch.no_grad():
                im_mat = get_sample(x, "latent")
                pred_mat = get_sample(pred_y, "pred")
                gt_mat = get_sample(y, "gt")
                
                dsc = util.dice(pred_mat, gt_mat)
                
                pbar.update(1)
                pbar.set_description('Epoch: %d. seg-loss: %.4f. Dice: %.4f.'
                                     %(epoch+1, losses.item(), dsc))
                # visualization
                if step % (len(train_loader)-1) == 0 and step != 0:
                    
                    pred_color = util.colorseg(gt_mat, pred_mat)
                    
                    fig, ax = plt.subplots(1,2,figsize=(8,4))
                    ax[0].imshow(im_mat, cmap="gray"),ax[0].axis("off")
                    ax[1].imshow(pred_color),ax[1].axis("off")
                    
                    fig.suptitle("train visual")
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.93)   
                    plt.show()
                    
    scheduler.step()
    
name = 'VFT_wo_optimization.pt'
torch.save(model.state_dict(),model_root+name)


    



