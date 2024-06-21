# coding:utf-8
# using pytorch implement dark channel prior algorithms, author: Mr zhipan wang
import numpy as np
import math
import cv2
import torch
from skimage import io
from module.guided_filter import FastGuidedFilter2d, GuidedFilter2d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def DarkChannel(im):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    return dc


def AtmLight(im,dark):
    """
    Calculate "A" in DCP algorithem
    """
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def DarkIcA(im, A):
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / (A[0, ind] + 0.0001)
    return DarkChannel(im3)


def Guidedfilter(im,p,r,eps):
    """
    guide filter
    """
    mean_I = cv2.boxFilter(im, -1, (r,r));   
    mean_p = cv2.boxFilter(p, -1, (r,r));
    mean_Ip = cv2.boxFilter(im*p, -1, (r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;


def TransmissionRefine_torch(im, et, fast=False):
    """
    transmission map
    """
    gray = im[:,:,0] * 0.2989 + im[:,:,1] * 0.5870 + im[:,:,2] * 0.1140
    
    r = 60;
    eps = 0.0001;
    
    gray = gray.unsqueeze(dim=0)
    gray = gray.unsqueeze(dim=1)
    
    et = et.unsqueeze(dim=0)
    et = et.unsqueeze(dim=1)
    
    if fast:
        GF = FastGuidedFilter2d(r, eps, 2)   
    else:
        GF = GuidedFilter2d(r, eps)
    
    t = GF(et, gray)   
    
    t = t.squeeze()

    return t



def hazeRemove(Img, w=0.1):
    """
    haze remove based on dark channel prior
    """
    im = Img.cpu().detach().numpy() 
    
    im = np.transpose(im, [1, 2, 0])
      
    dark = DarkChannel(im)
    defog_A = AtmLight(im, dark)
    IcA = DarkIcA(im, defog_A)
    
    IcA = torch.from_numpy(IcA).to(device)
    tx = 1 - w*IcA   
    

    im = torch.from_numpy(im).to(device)    
    tx = TransmissionRefine_torch(im, tx, fast=False)
    
    _,_,channel = im.shape

    
    for i in range(0, channel):
        im[:,:,i] = (im[:,:,i]-defog_A[0, i])/(torch.maximum(tx, torch.tensor(0.01))) + defog_A[0, i]
    
    return im