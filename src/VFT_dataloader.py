# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 04:51:03 2022

@author: hudew
"""


import numpy as np
import torch
import torch.utils.data as Data


class get_vft_dataset(Data.Dataset):
    
    def __init__(self, im_size, num_sample, data):
        super(get_vft_dataset, self).__init__()
        
        self.im_size = im_size
        self.num_sample = num_sample
        self.im = []
        self.gt = []
        self.vf = []
        
        for i in range(len(data)):
            im, vf, gt = data[i]
            im_patch, vf_patch, gt_patch = self.sample(im, vf, gt, self.num_sample, 
                                                       self.im_size)
            self.im += im_patch
            self.vf += vf_patch
            self.gt += gt_patch
            
    
    def __len__(self,):
        return len(self.im)
    
    
    def __getitem__(self, idx):
        x1 = self.im[idx]
        x2 = self.vf[idx]
        y = self.gt[idx]
        x1_tensor = torch.tensor(x1).type(torch.FloatTensor)
        x2_tensor = torch.tensor(x2).type(torch.FloatTensor)
        y_tensor = torch.tensor(y).type(torch.int64)
        return x1_tensor, x2_tensor, y_tensor
    
    # randomly sample num_sample patches from a image
    def sample(self, im, vf, gt, num_sample, psize):
        dim = im.shape
        sample_x = np.random.randint(0, dim[-2]-psize[0], num_sample)
        sample_y = np.random.randint(0, dim[-1]-psize[0], num_sample)
        
        im_patch = []
        vf_patch = []
        gt_patch = []
        
        for i in range(num_sample):
            px1 = im[sample_x[i]:sample_x[i]+psize[0], sample_y[i]:sample_y[i]+psize[1]]
            im_patch.append(px1[None,:,:])

            px2 = vf[:,sample_x[i]:sample_x[i]+psize[0], sample_y[i]:sample_y[i]+psize[1]]
            vf_patch.append(px2)

            py = gt[sample_x[i]:sample_x[i]+psize[0],sample_y[i]:sample_y[i]+psize[1]]
            gt_patch.append(py)
            
        return im_patch, vf_patch, gt_patch
    

def load_vf_dataset(im_size, num_sample, data, batch_size):
    dataset = get_vft_dataset(im_size, num_sample, data)
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


