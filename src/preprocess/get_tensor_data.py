# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:44:58 2022

@author: hudew
"""

import sys
sys.path.insert(0,"E:\\static representation\\src\\")
sys.path.insert(0,"E:\\tools\\")

import util
import frangi2d as fg
import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm

data_root = "E:\\static representation\\data\\"

with open(data_root+"raw_data.pickle","rb") as handle:
    data = pickle.load(handle)
    
data_tensor = {}

#%%



#%% tensor data
for key in list(data):
    data_tensor[key] = []
    
    for i in range(len(data[key])):
        im = data[key][i]
        name = key[:key.find("_")]
        
        if name in ["fa","rose","octa500"]:
            im = im.max()-im
        
        if key.endswith("im"):
            if key.startswith("fa") or key.startswith("hrf"):
                sigma2 = 2.5
            else:
                sigma2 = 1.5
                
            Ix, Iy, Lambda1, Lambda2 = fg.optimized_hessian(im, 0.5, sigma2, 0.5, True)
            mag1, mag2 = fg.normalize_scale(Lambda1, Lambda2, 1)
            mag2 *= im
            mag1 *= im.max() - im
            
            tensor = np.stack((Ix*mag1, Iy*mag1, -Iy*mag2, Ix*mag2), axis=0)
            data_tensor[key].append(tensor)
        else:
            data_tensor[key].append(im)


with open(data_root + "tensor_data.pickle","wb") as handle:
    pickle.dump(data_tensor, handle)
    
#%% add chase_db1
with open(data_root+"tensor_data.pickle","rb") as handle:
    tensor_data = pickle.load(handle)

tensor_data["chase_im"] = []
tensor_data["chase_gt"] = []

for i in range(len(data["chase_im"])):
    im = data["chase_im"][i]
    gt = data["chase_gt"][i]
    
    Ix, Iy, Lambda1, Lambda2 = fg.optimized_hessian(im, 0.5, 2.5, 0.5, True)
    mag1, mag2 = fg.normalize_scale(Lambda1, Lambda2, 1)
    mag2 *= im
    mag1 *= im.max() - im
    tensor = np.stack((Ix*mag1, Iy*mag1, -Iy*mag2, Ix*mag2), axis=0)
    
    tensor_data["chase_im"].append(tensor)
    tensor_data["chase_gt"].append(gt)
    
with open(data_root + "tensor_data.pickle","wb") as handle:
    pickle.dump(tensor_data, handle)
    

#%% vector data
def CLAHE(im, cl):
    im = np.uint8(im*255)
    clahe = cv2.createCLAHE(clipLimit = cl)
    opt = clahe.apply(im)
    return opt

for key in list(data):
    data_tensor[key] = []
    
    value = range(len(data[key]))
    with tqdm(total=len(value)) as pbar: 
    
        for i in range(len(data[key])):
            im = data[key][i]
            name = key[:key.find("_")]
            
            if name in ["fa","rose","octa500"]:
                im = im.max()-im
            else:
                im = np.float32(CLAHE(im, 7))
            
            if key.endswith("im"):
                if key.startswith("fa") or key.startswith("hrf"):
                    sigma2 = 2.5
                else:
                    sigma2 = 1.5
                    
                Ix, Iy, Lambda1, Lambda2 = fg.optimized_hessian(im, 0.5, sigma2, 0.5, True)
                mag = util.ImageRescale(im.max()-im, [0,255])
                
                vf = np.stack((Ix*mag, Iy*mag), axis=0)
                data_tensor[key].append(vf)
            else:
                data_tensor[key].append(im)
            
            pbar.update(1)
            pbar.set_description("dataset: {} item: {}".format(key,(i+1)))
            

with open(data_root + "vf_wo_augment.pickle","wb") as handle:
    pickle.dump(data_tensor, handle)
    
#%% VFT augmented training

namelist = ["octa500","rose"]
data_vf = {}
augment_data = {}
gt_list = []

for file in os.listdir(data_root):
    if file.endswith(".pickle") and file.startswith("augments(octa)"):
        name, _ = file.split(".")
        
        with open(data_root + file, "rb") as handle:
            augment_data[name] = pickle.load(handle)

with open(data_root + "raw_data.pickle", "rb") as handle:
    raw_data = pickle.load(handle)

for d in namelist:
    key = d + "_gt"
    gt_list += raw_data[key]
    

for key in list(augment_data):
    data_vf[key + "_im"] = []
    im_list = augment_data[key]
    
    value = range(len(im_list))
    with tqdm(total=len(value)) as pbar:
        
        for i in range(len(im_list)): 
            im = im_list[i]
            if key != "augments(octa)_4":
                im = im.max()-im
            
            Ix, Iy, Lambda1, Lambda2 = fg.optimized_hessian(im, 0.5, 1.5, 0.5, True)
            mag = util.ImageRescale(im.max()-im, [0,255])
            vf = np.stack((Ix*mag, Iy*mag), axis=0)
            
            pbar.update(1)
            pbar.set_description("dataset: {} item: {}".format(key,(i+1)))
            
            data_vf[key + "_im"].append(vf)
            data_vf[key + "_gt"] = gt_list


with open(data_root + "vf(octa)_augment.pickle","wb") as handle:
    pickle.dump(data_vf, handle)
    
        

