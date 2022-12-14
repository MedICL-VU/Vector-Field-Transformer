# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 10:04:30 2022

@author: hudew
"""

import sys 
sys.path.insert(0,'E:\\static representation\\baselines\\VFT\\')
sys.path.insert(0, "E:\\static representation\\src\\")
sys.path.insert(0, "E:\\tools\\")

import VFT_arch as net
import util
import models
import test_modules as tm

import numpy as np
import pickle
from tqdm import tqdm
import torch

# load model
model_root = "E:\\Model\\"
data_root = "E:\\static representation\\data\\"
save_root = "E:\\vector field learning\\ablation study\\result\\"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = net.VFTransformer_v3([2,16,32,64,64]).to(device)
model.load_state_dict(torch.load(model_root + 'VFT_wo_opimization.pt'))

#model = models.res_UNet([8,16,16,32,128], 2, 2).to(device)
#model.load_state_dict(torch.load(model_root + 'rUNet_on_vf.pt'))

#%% load data
with open(data_root+"raw_data.pickle", "rb") as handle:
    im_data = pickle.load(handle)
    
with open(data_root + "vf_wo_augment.pickle","rb") as handle:
    vf_data = pickle.load(handle)
    
test_dataset = "rose"
im_list = im_data[test_dataset + "_im"]
vf_list = vf_data[test_dataset + "_im"]
gt_list = im_data[test_dataset + "_gt"]

del im_data, vf_data

#%% test
dice_dict = {"VFT":[]}
test_method = "full" 
patch_size = [512,512]
stride = 50

values = range(len(im_list))
with tqdm(total=len(values)) as pbar:
    
    for i in range(len(im_list)):
        if test_dataset.startswith("octa") or test_dataset.startswith("rose") \
        or test_dataset.startswith("fa"):
            im = im_list[i]
        else:
            im = tm.reverse_int(im_list[i])
        vf = vf_list[i]
        gt = gt_list[i]
        
        if test_method == "full":
            test = tm.Size_Adaptive_Test(im, vf)
            pred_y = test.single_model_test(test.x2, model, device)
        elif test_method == "patch":
            test = tm.Patch_Pad_Test(im, vf, patch_size, stride)
            pred_y = test.single_model_test(test.tf_patch_split, model, device)
        else:
            raise ValueError
        
        dice_dict["VFT"].append(util.dice(pred_y,gt))
        pbar.update(1)

with open(save_root + "VFT_optimized_wo_aug_{}.pickle".format(test_dataset),"wb") as handle:
    pickle.dump(dice_dict, handle)
    
