# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:52:05 2022

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\static representation\\baselines\\VFT\\')
sys.path.insert(0, "E:\\static representation\\src\\")
sys.path.insert(0,'E:\\tools\\')

import util
import augment_arch as arch
import test_modules as tm

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pickle

model_root = "E:\\Model\\"
data_root = "E:\\static representation\\data\\"
save_root = "E:\\static representation\\baselines\\"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

enh_enc = (8,16,32,64,64)
seg_enc = (8,16,32)
nch_in = 1
nch_out = 1

model = arch.VAE(enh_enc, seg_enc, nch_in, nch_out).to(device)
model.load_state_dict(torch.load(model_root+"VFT_augment(octa)_3.pt"))

#%% load data
with open(data_root+"raw_data.pickle", "rb") as handle:
    im_data = pickle.load(handle)
        
test_datasets = ["octa500","rose"]

im_list = []
for i in range(len(test_datasets)):
    im_list += im_data[test_datasets[i] + "_im"]

del im_data

#%% test
patch_size = [256,256]
stride = 1
latent_list = []

for i in range(len(im_list)):
    im = im_list[i]
    
    h, w = im.shape
    if h == 400:
        patch_size = [384,384]
    else:
        patch_size = [256,256]
    
    test = tm.latent_reconstruct(im, patch_size, stride)
    pred_recon = test.reconstruct(model, device)
    latent_list.append(pred_recon)
    
    fig,ax = plt.subplots(1,2,figsize=(8,4))
#    ax[0].imshow(np.transpose(im,(1,2,0))),ax[0].axis("off")
    ax[0].imshow(im, cmap='gray'),ax[0].axis("off")
    ax[1].imshow(pred_recon, cmap='gray'),ax[1].axis("off")
    
    fig.suptitle("idx={}".format(i+1))
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.show()
    
with open(save_root + "augments(octa)_3.pickle", "wb") as handle:
    pickle.dump(latent_list, handle)
    