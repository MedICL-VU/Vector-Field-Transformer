# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 00:24:47 2022

@author: hudew
"""

import numpy as np
from scipy.ndimage.filters import convolve


def Hessian2D(I, Sigma=1):
    """
    This function Hessian2 filters the image with 2nd derivatives of a
    Gaussian with parameter Sigma.
    :param I: image, in flotaing point precision (float64)
    :param Sigma: sigma of the gaussian kernel used
    :return: the 2nd derivatives
    """
    # Make kernel coordinates
    X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1),
                       np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

    # Build the gaussian 2nd derivatives filters
    DGaussxx = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
    DGaussxy = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
    DGaussyy = DGaussxx.conj().T

    Dxx = convolve(I, DGaussxx, mode='constant', cval=0.0)
    Dxy = convolve(I, DGaussxy, mode='constant', cval=0.0)
    Dyy = convolve(I, DGaussyy, mode='constant', cval=0.0)

    return Dxx, Dxy, Dyy


def eig2image(Dxx, Dxy, Dyy):
    """
    This function eig2image calculates the eigen values from the
    hessian matrix, sorted by abs value. And gives the direction
    of the ridge (eigenvector smallest eigenvalue) .
    | Dxx  Dxy |
    | Dxy  Dyy |
    """
    # Compute the eigenvectors of J, v1 and v2
    tmp = np.sqrt((Dxx - Dyy)**2 + 4*Dxy**2)
    v2x = 2*Dxy
    v2y = Dyy - Dxx + tmp

    # Normalize
    mag = np.sqrt(v2x**2 + v2y**2)
    i = np.invert(np.isclose(mag, np.zeros(mag.shape)))
    v2x[i] = v2x[i]/mag[i]
    v2y[i] = v2y[i]/mag[i]

    # The eigenvectors are orthogonal
    v1x = -v2y.copy()
    v1y = v2x.copy()

    # Compute the eigenvalues
    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    # Sort eigenvalues by absolute value abs(Lambda1)<abs(Lambda2)
    check = np.absolute(mu1) > np.absolute(mu2)

    Lambda1 = mu1.copy()
    Lambda1[check] = mu2[check]
    Lambda2 = mu2.copy()
    Lambda2[check] = mu1[check]

    Ix = v1x.copy()
    Ix[check] = v2x[check]
    Iy = v1y.copy()
    Iy[check] = v2y[check]

    return Lambda1, Lambda2, Ix, Iy


# --------------------------------------------------------


def FrangiFilter2D(I, FrangiScaleRange=np.array([1, 10]), FrangiScaleRatio=2,
                   FrangiBetaOne=0.5, FrangiBetaTwo=15, verbose=False, BlackWhite=True):
    """
    This function FRANGIFILTER2D uses the eigenvectors of the Hessian to
    compute the likeliness of an image region to vessels, according
    to the method described by Frangi:2001 (Chapter 2). Adapted from MATLAB code
    :param I: imput image (grayscale)
    :param FrangiScaleRange: The range of sigmas used, default [1 10]
    :param FrangiScaleRatio: Step size between sigmas, default 2
    :param FrangiBetaOne: Frangi correction constant, default 0.5
    :param FrangiBetaTwo: Frangi correction constant, default 15
    :param verbose: Show debug information, default false
    :param BlackWhite: Detect black ridges (default) set to true, for white ridges set to false.
    :return: The vessel enhanced image (pixel is the maximum found in all scales)
    """

    if len(FrangiScaleRange) > 1:
        sigmas = np.arange(FrangiScaleRange[0], FrangiScaleRange[1]+1, FrangiScaleRatio)
        sigmas = sorted(sigmas)
    else:
        sigmas = [FrangiScaleRange[0]]
    beta = 2*FrangiBetaOne**2
    c = 2*FrangiBetaTwo**2

    # Make matrices to store all filterd images
    ALLfiltered = np.zeros([I.shape[0], I.shape[1], len(sigmas)])
    ALLangles = np.zeros([I.shape[0], I.shape[1], len(sigmas)])

    # Frangi filter for all sigmas
    for i in range(len(sigmas)):
        # Show progress
        if verbose:
            print('Current Frangi Filter Sigma: ', str(sigmas[i]))

        # Make 2D hessian
        Dxx, Dxy, Dyy = Hessian2D(I, sigmas[i])

        # Correct for scale
        Dxx *= (sigmas[i]**2)
        Dxy *= (sigmas[i]**2)
        Dyy *= (sigmas[i]**2)

        # Calculate (abs sorted) eigenvalues and vectors
        Lambda2, Lambda1, Ix, Iy = eig2image(Dxx, Dxy, Dyy)

        # Compute the direction of the minor eigenvector
        angles = np.arctan2(Ix, Iy)

        # Compute some similarity measures
        near_zeros = np.isclose(Lambda1, np.zeros(Lambda1.shape))
        Lambda1[near_zeros] = 2**(-52)
        Rb = (Lambda2/Lambda1)**2
        S2 = Lambda1**2 + Lambda2**2

        # Compute the output image
        Ifiltered = np.exp(-Rb/beta)*(np.ones(I.shape)-np.exp(-S2/c))

        # see pp. 45
        if BlackWhite:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        # store the results in 3D matrices
        ALLfiltered[:, :, i] = Ifiltered.copy()
        ALLangles[:, :, i] = angles.copy()


    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value
    if len(sigmas) > 1:
        outIm = np.amax(ALLfiltered, axis=2)
        outIm = outIm.reshape(I.shape[0], I.shape[1], order='F')
        whatScale = np.argmax(ALLfiltered, axis=2)
        whatScale = np.reshape(whatScale, I.shape, order='F')

        indices = range(I.size) + (whatScale.flatten(order='F') - 1)*I.size
        values = np.take(ALLangles.flatten(order='F'), indices)
        direction = np.reshape(values, I.shape, order='F')
    else:
        outIm = ALLfiltered.reshape(I.shape[0], I.shape[1], order='F')
        whatScale = np.ones(I.shape)
        direction = np.reshape(ALLangles, I.shape, order='F')

    return outIm, whatScale, direction


def optimized_hessian(im, sigma_1, sigma_2, step, bw):
    sigmas = np.arange(sigma_1, sigma_2+1, step)
    sigmas = sorted(sigmas)
    
    h,w = im.shape
    all_filtered_Ix = np.zeros([len(sigmas),h,w],dtype=np.float32)
    all_filtered_Iy = np.zeros([len(sigmas),h,w],dtype=np.float32)
    all_filtered_lambda1 = np.zeros([len(sigmas),h,w],dtype=np.float32)
    all_filtered_lambda2 = np.zeros([len(sigmas),h,w],dtype=np.float32)
    
    opt,scale,d = FrangiFilter2D(np.float32(im),np.array([sigma_1,sigma_2]),step,0.5,15,False,bw)
    
    for i in range(len(sigmas)):
        # Make 2D hessian
        Dxx, Dxy, Dyy = Hessian2D(im, sigmas[i])
    
        # Correct for scale
        Dxx *= (sigmas[i]**2)
        Dxy *= (sigmas[i]**2)
        Dyy *= (sigmas[i]**2)
    
        # Calculate (abs sorted) eigenvalues and vectors
        all_filtered_lambda1[i,:,:], all_filtered_lambda2[i,:,:], \
        all_filtered_Ix[i,:,:], all_filtered_Iy[i,:,:] = eig2image(Dxx, Dxy, Dyy)
    
    Ix = np.zeros([h,w],dtype=np.float32)
    Iy = np.zeros([h,w],dtype=np.float32)
    Lambda1 = np.zeros([h,w],dtype=np.float32)
    Lambda2 = np.zeros([h,w],dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            idx = scale[i,j]
            Ix[i,j] = all_filtered_Ix[idx,i,j]
            Iy[i,j] = all_filtered_Iy[idx,i,j]
            Lambda1[i,j] = all_filtered_lambda1[idx,i,j]
            Lambda2[i,j] = all_filtered_lambda2[idx,i,j]
            
    return Ix, Iy, Lambda1, Lambda2


def normalize_scale(lambda1, lambda2, alpha):
    f_norm = lambda1**2 + lambda2**2
    mag1 = np.exp(-alpha*lambda1**2/(f_norm+0.001))
    mag2 = np.exp(-alpha*lambda2**2/(f_norm+0.001))
    return mag1, mag2