# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:51:11 2021

@author: hudew
"""

import torch
import torch.nn as nn

class Residual_block(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Residual_block,self).__init__()
        self.align = nn.Conv2d(in_channels = nch_in,
                               out_channels = nch_out,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0)
        self.dualconv = nn.Sequential(
                nn.Conv2d(in_channels = nch_out,
                          out_channels = nch_out,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features = nch_out),
                nn.ELU(),
                nn.Conv2d(in_channels = nch_out,
                          out_channels = nch_out,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features = nch_out)
                )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.align(x)
        x1 = self.dualconv(x)
        opt = self.relu(torch.add(x,x1))
        return opt
    
    
def trans_down(nch_in, nch_out):
    return nn.Sequential(
            nn.Conv2d(in_channels=nch_in, 
                      out_channels=nch_out, 
                      kernel_size=4,
                      stride=2, 
                      padding=1),
            nn.BatchNorm2d(nch_out),
            nn.ELU()
            )
            
            
def trans_up(nch_in,nch_out):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=nch_in, 
                               out_channels=nch_out,
                               kernel_size=4, 
                               stride=2, 
                               padding=1),
            nn.BatchNorm2d(nch_out),
            nn.ELU()
            )
            
class res_UNet(nn.Module):
    def __init__(self, nch_enc, nch_in, nch_out):
        super(res_UNet,self).__init__()
        
        # (assume input_channel=1)
        self.nch_in = nch_in
        self.nch_enc = nch_enc
        self.nch_aug = (self.nch_in,)+self.nch_enc
        
        # module list
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        
        for i in range(len(self.nch_enc)):
            # encoder & downsample
            self.encoder.append(Residual_block(self.nch_aug[i], self.nch_aug[i+1]))
            self.td.append(trans_down(self.nch_enc[i], self.nch_enc[i]))
            # decoder & upsample
            self.tu.append(trans_up(self.nch_enc[-1-i], self.nch_enc[-1-i]))
            if i == len(self.nch_enc)-1:
                self.decoder.append(Residual_block(self.nch_aug[-1-i]*2, nch_out))
            else:
                self.decoder.append(Residual_block(self.nch_aug[-1-i]*2, self.nch_aug[-2-i]))
    
    
    def forward(self, x):
        cats = []
        # encoder
        for i in range(len(self.nch_enc)):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom layer
        layer_opt = x
        
        # decoder
        for i in range(len(self.nch_enc)):
            x = self.tu[i](layer_opt)
            x = torch.cat([x,cats[-1-i]],dim=1)
            layer_opt = self.decoder[i](x)

        y_pred = layer_opt
        return y_pred


class VAE(nn.Module):
    def __init__(self, enh_enc, seg_enc, nch_in, nch_out):
        super(VAE, self).__init__()
        
        self.enh_enc = enh_enc
        self.seg_enc = seg_enc
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.bifurcate = nn.Conv2d(nch_out,2,1,1,0)
        
        self.Enh_Net = res_UNet(self.enh_enc, self.nch_in, self.nch_out)
        self.Seg_Net = res_UNet(self.seg_enc, 1, 2)
    
    def reparameterize(self, seg):
        latent = self.bifurcate(seg)
        mu = torch.unsqueeze(latent[:,0,:,:],dim=1)
        log_var = torch.unsqueeze(latent[:,1,:,:],dim=1)
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
#        sample = self.sigmoid(torch.log(eps)-torch.log(1-eps)+\
#                              torch.log(seg)-torch.log(1-seg))
        sample = mu+(eps*std)
        return sample
    
    def forward(self, x):
        # Encoder
        Enh_opt = self.Enh_Net(x)           # [batch,channel=1,H,W]
        # Reparameterize
        z = self.reparameterize(Enh_opt)
        # Decoder
        Seg_opt = self.Seg_Net(z)        
        return Enh_opt, Seg_opt
        
        
    