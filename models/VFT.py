# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:10:42 2021

@author: hudew
"""

import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

#%% Basic blocks
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    assert (H % window_size == 0 and W % window_size == 0), "Invalid window_size."
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, H, W)
    return x


class Attention(nn.Module):
    '''
    input: [B, N, C] 
    '''
    def __init__(self, num_heads, hidden_size, dropout):
        super(Attention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_size = hidden_size 
        self.dropout = dropout
        self.head_size = int(self.hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.head_size
        
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        
        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)
        
        self.softmax = nn.Softmax(dim=-1)
    
    
    def transpose_for_scores(self, x):
        # x: [B, N, C] -> [B, N, num_heads, head_size] 
        #              -> [B, num_heads, N, head_size] 
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    
    def forward(self, hidden_state):
        q_expend = self.query(hidden_state)
        k_expend = self.key(hidden_state)
        v_expend = self.value(hidden_state)
        
        q = self.transpose_for_scores(q_expend)
        k = self.transpose_for_scores(k_expend)
        v = self.transpose_for_scores(v_expend)
        
        attn = q @ (k.transpose(-1,-2))
        attn = attn / math.sqrt(self.head_size)
        attn_prob = self.softmax(attn)
        attn_prob = self.attn_dropout(attn_prob)
        
        context = attn_prob @ v
        context = context.permute(0, 2, 1, 3).contiguous() # [B, N, num_heads, head_size]
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)         # [B, N, C]
        opt = self.out(context)
        opt = self.proj_dropout(opt)
        
        return opt


class Mlp(nn.Module):
    
    def __init__(self, hidden_size, mlp_dim, dropout):
        super(Mlp, self).__init__()

        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        
        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Projection(nn.Module):
    
    def __init__(self, nch_in, nch_out, dropout=0.):
        super(Projection, self).__init__()
        
        self.nch_in = nch_in
        self.nch_out = nch_out
        
        self.fc = nn.Linear(self.nch_in, self.nch_out)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=1e-6)


    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = F.normalize(x, dim=1)
        return x


class Transblock(nn.Module):
    '''
    input x: [Nw*B, Ws**2, C]
    output : [Nw*B, Ws**2, C]
    '''
    def __init__(self, num_heads, hidden_size, mlp_dim, dropout):
        super(Transblock, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        
        self.attn = Attention(self.num_heads, self.hidden_size, dropout)
        self.mlp = Mlp(self.hidden_size, self.mlp_dim, dropout)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        nB, N, C = x.shape
        
        buffer_1 = x
        x = self.ln(x)
        x = self.attn(x)
        x = self.dropout(x) + buffer_1
        
        buffer_2 = x
        x = self.ln(x)
        x = self.mlp(x)
        opt = self.dropout(x) + buffer_2
        
        return opt
    

class Layer(nn.Module):
    '''
    input x: [B, C, H, W]
    output : [B, C, H, W]
    '''
    def __init__(self, window_size, n_block, num_heads, hidden_size, mlp_dim, dropout):
        super(Layer, self).__init__()
        
        self.window_size = window_size
        self.n_block = n_block        
        self.transblocks = nn.ModuleList([
                Transblock(num_heads, hidden_size, mlp_dim, dropout)
                for i in range(self.n_block)])
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # window partition
        x = window_partition(x, self.window_size)  # [Nw*B, Ws, Ws, C]
        nB, Ws, Ws, C = x.shape
        x = x.reshape(nB, Ws*Ws, C)                # [Nw*B, Ws*Ws, C]
        
        # transblocks
        for t_block in self.transblocks:
            x = t_block(x)
        x = x.reshape(nB, Ws, Ws, C)
        
        # reverse
        x = window_reverse(x, self.window_size, H, W)
        
        return x

class Multi_Wsize_Layer(nn.Module):
    
    def __init__(self, num_heads, hidden_size, mlp_dim, dropout):
        super(Multi_Wsize_Layer, self).__init__()
        
        self.layer_w2 = Layer(window_size = 2,
                              n_block = 1,
                              num_heads = num_heads,
                              hidden_size = hidden_size,
                              mlp_dim = mlp_dim,
                              dropout = dropout)
        
        self.layer_w4 = Layer(window_size = 4,
                              n_block = 1,
                              num_heads = num_heads,
                              hidden_size = hidden_size,
                              mlp_dim = mlp_dim,
                              dropout = dropout)
        
        self.layer_w8 = Layer(window_size = 8,
                              n_block = 1,
                              num_heads = num_heads,
                              hidden_size = hidden_size,
                              mlp_dim = mlp_dim,
                              dropout = dropout)
    
        self.combine = nn.Conv2d(in_channels = int(hidden_size*3),
                                 out_channels = hidden_size,
                                 kernel_size = 3,
                                 stride = 1,
                                 padding = 1)
        
    def forward(self, x):
        x = torch.cat((self.layer_w2(x),self.layer_w4(x),self.layer_w8(x)),1)
        opt = self.combine(x)
        return opt
    
    
class Multi_Wsize_Layer_v2(nn.Module):
    
    def __init__(self, num_heads, hidden_size, mlp_dim, dropout):
        super(Multi_Wsize_Layer_v2, self).__init__()
        
        self.layer_w2 = Layer(window_size = 2,
                              n_block = 1,
                              num_heads = num_heads,
                              hidden_size = hidden_size,
                              mlp_dim = mlp_dim,
                              dropout = dropout)
        
        self.layer_w4 = Layer(window_size = 4,
                              n_block = 1,
                              num_heads = num_heads,
                              hidden_size = hidden_size,
                              mlp_dim = mlp_dim,
                              dropout = dropout)
        
        self.layer_w8 = Layer(window_size = 8,
                              n_block = 1,
                              num_heads = num_heads,
                              hidden_size = hidden_size,
                              mlp_dim = mlp_dim,
                              dropout = dropout)
    
        self.combine = nn.Sequential(
                                nn.Conv2d(in_channels = int(hidden_size*3),
                                          out_channels = hidden_size,
                                          kernel_size = 3,
                                          stride = 1,
                                          padding = 1),
                                nn.BatchNorm2d(hidden_size),
                                nn.ReLU(inplace=True)
                                )
    def forward(self, x):
        x = torch.cat((self.layer_w2(x),self.layer_w4(x),self.layer_w8(x)),1)
        opt = self.combine(x)
        return opt


class Recurrent_block(nn.Module):
    def __init__(self, nch, t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.nch = nch
        self.conv = nn.Sequential(
                nn.Conv2d(nch,nch,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(nch),
                nn.ReLU(inplace=True)
                )
    
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1
    
    
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


def trans_down(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )


def trans_up(in_channels, out_channels):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )
                
#%%
class res_UNet(nn.Module):
    def __init__(self, nch_in, nch_enc):
        super(res_UNet, self).__init__()
    
        self.nch_enc = nch_enc
        self.nch_aug = nch_enc[:]
        self.nch_aug.insert(0, nch_in)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        
        for i in range(len(self.nch_enc)):
            # encoder & downsample
            self.encoder.append(Residual_block(self.nch_aug[i],self.nch_aug[i+1]))
            self.td.append(trans_down(self.nch_enc[i],self.nch_enc[i]))
            # decoder & upsample
            self.tu.append(trans_up(self.nch_enc[-1-i],self.nch_enc[-1-i]))
            if i == len(self.nch_enc)-1:
                self.decoder.append(Residual_block(self.nch_aug[-1-i]*2,2))
            else:
                self.decoder.append(Residual_block(self.nch_aug[-1-i]*2,self.nch_aug[-2-i]))
    
    def forward(self, x):
        cats = []
        # encoder
        for i in range(len(self.nch_enc)):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom layer
        latent = x
        layer_opt = x
        
        # decoder
        for i in range(len(self.nch_enc)):
            x = self.tu[i](layer_opt)
            x = torch.cat([x,cats[-1-i]],dim=1)
            layer_opt = self.decoder[i](x)

        y_pred = layer_opt
        return y_pred, latent
    
    
class VFTransformer(nn.Module):
    
    def __init__(self, nch_enc, n_blocks, window_sizes):
        super(VFTransformer, self).__init__()
        
        self.nch_enc = nch_enc
        self.nch_dec = nch_enc[::-1]
        self.n_blocks = n_blocks
        assert len(self.nch_enc) == len(self.n_blocks), "The length of the two list should equal"
        
        self.window_sizes = window_sizes
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
    
        for i in range(len(self.nch_enc)):
            self.encoder.append(Layer(window_size = self.window_sizes[i],
                                      n_block = self.n_blocks[i],
                                      num_heads = self.nch_enc[i] // 2,# set the head_size to be 2
                                      hidden_size = self.nch_enc[i],
                                      mlp_dim = self.nch_enc[i] * 4,
                                      dropout = 0.)
                                )
            if i == 0:
                self.decoder.append(Residual_block(self.nch_dec[i], self.nch_dec[i+1]))
            elif i == len(self.nch_enc)-1:
                self.decoder.append(Residual_block(self.nch_dec[i]*2, self.nch_enc[0]))
            else:
                self.decoder.append(Residual_block(self.nch_dec[i]*2, self.nch_dec[i+1]))
            
            if i < len(self.nch_enc)-1:
                self.td.append(trans_down(self.nch_enc[i], self.nch_enc[i+1]))
                self.tu.append(trans_up(self.nch_dec[i+1], self.nch_dec[i+1]))
        
    def forward(self, x):
        cats = []
        
        # encoder
        for i in range(len(self.nch_enc)-1):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom
        latent = self.encoder[-1](x)
        layer_opt = self.decoder[0](latent)
        
        # decoder
        for i in range(len(self.nch_dec)-1):
            x = self.tu[i](layer_opt)
            x = torch.cat([x, cats[-1-i]],dim=1)
            layer_opt = self.decoder[i+1](x)
        
        y_pred = layer_opt
        return y_pred, latent             
        

class VFTransformer_v2(nn.Module):
    
    def __init__(self, nch_enc):
        super(VFTransformer_v2, self).__init__()
        self.nch_enc = nch_enc
        self.nch_dec = nch_enc[::-1]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        
        for i in range(len(self.nch_enc)):
            self.encoder.append(Multi_Wsize_Layer(num_heads = self.nch_enc[i] // 2,
                                                  hidden_size = self.nch_enc[i],
                                                  mlp_dim = self.nch_enc[i] * 4,
                                                  dropout = 0.))
            if i == 0:
                self.decoder.append(Residual_block(self.nch_dec[i], self.nch_dec[i+1]))
            elif i == len(self.nch_enc)-1:
                self.decoder.append(Residual_block(self.nch_dec[i]*2, self.nch_enc[0]))
            else:
                self.decoder.append(Residual_block(self.nch_dec[i]*2, self.nch_dec[i+1]))
            
            if i < len(self.nch_enc)-1:
                self.td.append(trans_down(self.nch_enc[i], self.nch_enc[i+1]))
                self.tu.append(trans_up(self.nch_dec[i+1], self.nch_dec[i+1])) 
    
    def forward(self, x):
        cats = []
        
        # encoder
        for i in range(len(self.nch_enc)-1):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom
        latent = self.encoder[-1](x)
        layer_opt = self.decoder[0](latent)
        
        # decoder
        for i in range(len(self.nch_dec)-1):
            x = self.tu[i](layer_opt)
            x = torch.cat([x, cats[-1-i]],dim=1)
            layer_opt = self.decoder[i+1](x)
        
        y_pred = layer_opt
        return y_pred, latent


class VFTransformer_v3(nn.Module):
    
    def __init__(self, nch_enc):
        super(VFTransformer_v3, self).__init__()
        self.nch_enc = nch_enc
        self.nch_dec = nch_enc[::-1]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.td = nn.ModuleList()
        self.tu = nn.ModuleList()
        
        for i in range(len(self.nch_enc)):
            self.encoder.append(Multi_Wsize_Layer_v2(num_heads = self.nch_enc[i] // 2,
                                                  hidden_size = self.nch_enc[i],
                                                  mlp_dim = self.nch_enc[i] * 4,
                                                  dropout = 0.))
            if i == 0:
                self.decoder.append(Residual_block(self.nch_dec[i], self.nch_dec[i+1]))
            elif i == len(self.nch_enc)-1:
                self.decoder.append(Residual_block(self.nch_dec[i]*2, 2))
            else:
                self.decoder.append(Residual_block(self.nch_dec[i]*2, self.nch_dec[i+1]))
            
            if i < len(self.nch_enc)-1:
                self.td.append(trans_down(self.nch_enc[i], self.nch_enc[i+1]))
                self.tu.append(trans_up(self.nch_dec[i+1], self.nch_dec[i+1])) 
    
    def forward(self, x):
        cats = []
        
        # encoder
        for i in range(len(self.nch_enc)-1):
            layer_opt = self.encoder[i](x)
            x = self.td[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom
        latent = self.encoder[-1](x)
        layer_opt = self.decoder[0](latent)
        
        # decoder
        for i in range(len(self.nch_dec)-1):
            x = self.tu[i](layer_opt)
            x = torch.cat([x, cats[-1-i]],dim=1)
            layer_opt = self.decoder[i+1](x)
        
        y_pred = layer_opt
        return y_pred, latent



class ContrastiveVFT(nn.Module):
    def __init__(self, nch_enc, nch_ipt1, nch_ipt2):
        super(ContrastiveVFT, self).__init__()
        
        self.nch_enc = nch_enc
        
        self.vft_enc = nch_enc[:]
        self.vft_enc.insert(0, nch_ipt1)
        self.vft_dec = self.vft_enc[::-1]
        
        self.cnn_enc = nch_enc[:]
        self.cnn_enc.insert(0, nch_ipt2)
        
        self.cnn_encoder = nn.ModuleList()
        self.cnn_td = nn.ModuleList()
        
        self.vft_encoder = nn.ModuleList()
        self.vft_decoder = nn.ModuleList()
        self.vft_td = nn.ModuleList()
        self.tu = nn.ModuleList()
        
        # VFT that include encoder 1 and decoder
        for i in range(len(self.vft_enc)):
            self.vft_encoder.append(Multi_Wsize_Layer(num_heads = self.vft_enc[i] // 2,
                                                  hidden_size = self.vft_enc[i],
                                                  mlp_dim = self.vft_enc[i] * 4,
                                                  dropout = 0.))
            if i == 0:
                self.vft_decoder.append(Residual_block(self.vft_dec[i], self.vft_dec[i+1]))
            elif i == len(self.nch_enc):
                self.vft_decoder.append(Residual_block(self.vft_dec[i]*2, self.vft_enc[0]))
            else:
                self.vft_decoder.append(Residual_block(self.vft_dec[i]*2, self.vft_dec[i+1]))
            
            if i < len(self.nch_enc):
                self.vft_td.append(trans_down(self.vft_enc[i], self.vft_enc[i+1]))
                self.tu.append(trans_up(self.vft_dec[i+1], self.vft_dec[i+1]))
        
        # Encoder 2
        for i in range(len(self.nch_enc)):
            self.cnn_encoder.append(Residual_block(self.cnn_enc[i],self.cnn_enc[i+1]))
            self.cnn_td.append(trans_down(self.nch_enc[i],self.nch_enc[i]))
        
    def forward(self, x_vf, x_im):
        cats = []
        
        # VFT encoder
        for i in range(len(self.nch_enc)):
            layer_opt = self.vft_encoder[i](x_vf)
            x_vf = self.vft_td[i](layer_opt)
            cats.append(layer_opt)
        
        # CNN encoder
        for i in range(len(self.nch_enc)):
            opt = self.cnn_encoder[i](x_im)
            x_im = self.cnn_td[i](opt)
        
        # latent space
        l_vft = self.vft_encoder[-1](x_vf)
        l_cnn = x_im
        
        # decoder
        layer_opt = self.vft_decoder[0](l_vft)
        
        for i in range(len(self.nch_enc)):
            x_vf = self.tu[i](layer_opt)
            x_vf = torch.cat([x_vf, cats[-1-i]],dim=1)
            layer_opt = self.vft_decoder[i+1](x_vf)
        
        y_pred = layer_opt
        return y_pred, l_vft, l_cnn
        