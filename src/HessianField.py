# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 23:58:24 2021

@author: hudew
"""

import numpy as np
from skimage.feature import hessian_matrix

def abs_evalue(H,reorder):
    # evalues in ascending order (-h,-h,l)/(-h,l,l)
    # each column of evectors is normalized eigenvector 
    evalues,evectors = np.linalg.eigh(H)
    
    # just eigen-decomposition
    if reorder == 0:
        evalues = evalues
        evectors = evectors
    
    # take absolute value of eigenvalues (h,h,l)/(h,l,l)
    elif reorder == 1:
        evalues = np.abs(evalues)
        evectors = evectors
    
    # take absolute value of eigenvalues and swap the first and the third eigenvector
    # equivalent to swap lambda1 and lambda3 (l,h,h)/(l,l,h)
    elif reorder == 2:
        evalues = np.abs(evalues)
        evectors = np.fliplr(evectors)
    
    # use the flipping formula (l,l,h)/(l,h,h)
    elif reorder == 3:
        λ1,λ2,λ3 = evalues
        norm = np.sqrt(λ1**2+λ2**2+λ3**2)
        evalues = np.log(np.abs([1/λ1,1/λ2,1/λ3])*norm)*norm
    
    # x-z axis flip
    evectors = np.flipud(evectors)
    
    return evectors.dot(np.diag(evalues)).dot(evectors.T)


def symmetrize(elements):
    "complete the symmetric matrix given elements in upper triangle"
    if len(elements) == 6:
        # populate the upper triangle from vector
        opt = np.zeros((3,3))
        idx = np.triu_indices(len(opt))
        opt[idx] = elements
        # fill the lower trangle
        opt_diag = np.diag(np.diag(opt))
        opt = opt+opt.T-opt_diag
        
    elif len(elements) == 3:
        # populate the upper triangle from vector
        opt = np.zeros((2,2))
        idx = np.triu_indices(len(opt))
        opt[idx] = elements
        # fill the lower trangle
        opt_diag = np.diag(np.diag(opt))
        opt = opt+opt.T-opt_diag
    else:
        raise ValueError('Length of vector invalid.')
    
    return opt

        
def GetTensor(vol, sigma, reorder):
    h,w = vol.shape
    H_elements = np.array(hessian_matrix(vol,sigma,order='rc'))
    vec = H_elements.reshape((3,-1))

    _,num = vec.shape
    vec_tensor = np.zeros((4,num),dtype=np.float32)
    
    for i in range(num):
        elements = vec[:,i]
        # fill the Hessian
        H = symmetrize(elements)
        
        # reorder the eigenvalues to get tensor field
        tensor = abs_evalue(H,reorder)
        vec_tensor[:,i] = tensor.reshape(-1)
    
    return vec_tensor.reshape((4,h,w))


def GetFlow(im, sigma=0.01, scale=1):
    tensor = GetTensor(im, sigma, 2)
    
    l,h,w = tensor.shape
    flow = np.zeros([2,h,w],dtype=np.float32)
    flow_perp = np.zeros([2,h,w],dtype=np.float32)
        
    for i in range(h):
        for j in range(w):
            mat = tensor[:,i,j].reshape(2,2)
            e_val,e_vec = np.linalg.eigh(mat)
            
            if scale == 1:
                flow[:,i,j] = im[i,j] * e_vec[:,0]
                flow_perp[:,i,j] = im[i,j] * e_vec[:,1]
            elif scale == 2:
                flow[:,i,j] = e_val[0] * e_vec[:,0]
                flow_perp[:,i,j] = e_val[0] * e_vec[:,1]
            else:
                flow[:,i,j] = e_vec[:,0]
                flow_perp[:,i,j] = e_vec[:,1]
                
    return flow, flow_perp


def stack_Flow(im, sigma=0.01):
    tensor = GetTensor(im, sigma, 2)
    
    l,h,w = tensor.shape
    flow = np.zeros([3,h,w],dtype=np.float32)
        
    for i in range(h):
        for j in range(w):
            mat = tensor[:,i,j].reshape(2,2)
            e_val,e_vec = np.linalg.eigh(mat)
            flow[0,i,j] = im[i,j]
            flow[1:,i,j] = e_vec[:,0]
    return flow

#%%




        