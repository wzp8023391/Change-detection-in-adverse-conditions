# coding:utf-8
# using pytorch to implement image enhancement
# author:Mr zhipan wang, Email:1044625113@qq.com
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from easydict import EasyDict as edict
from module.dehaze_dcp import hazeRemove


cfg = edict()   
cfg.exposure_range = 3.5     
cfg.usm_range = (0.0, 5)
cfg.cont_range = (0.0, 1.0)
cfg.gamma_range = 3

cfg.defog_range = (0.03, 1.0)   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rgb2lum(image):
    """
    rgb image to lum space
    """
    image = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]
    image = image.unsqueeze(1)
    
    return image


def lerp(a, b, l):
    
    return (1 - l) * a + l * b


def tanh01(x):

  return torch.tanh(x) * 0.5 + 0.5   


def tanh_range(x, l, r, initial=None):
  """
  data norm
  """
    
  def get_activation(x, left, right, initial):

    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)   
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left   

    return activation(x)

  return get_activation(x, l, r, initial)


class ExposureFilter(nn.Module):
    
  def __init__(self, inputChannel):
    super(ExposureFilter, self).__init__()
    
    self.num_filter_parameters = 1
    self.inputChannel = inputChannel

  def forward(self, img, param):
    param = tanh_range(param, -cfg.exposure_range, cfg.exposure_range, initial=None)    # 0 
    
    img = img * torch.exp(param * np.log(2))
    
    return img


class UsmFilter(nn.Module):

  def __init__(self, inputChannel):
    super(UsmFilter, self).__init__()
    
    self.num_filter_parameters = 1
    self.channels = inputChannel

  def forward(self, img, param):
    param = tanh_range(param, *cfg.usm_range, initial=None)   # 3

    kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    
    kernel = np.repeat(kernel, self.channels, axis=0)

    kernel = kernel.to(img.device)

    output = F.conv2d(img, kernel, padding=2, groups=self.channels)

    img_out = (img - output) * param + img

    return img_out


class ContrastFilter(nn.Module):

  def __init__(self, inputChannel):
    super(ContrastFilter, self).__init__()
    
    self.num_filter_parameters = 1
    self.channels = inputChannel

  def forward(self, img, param):
    param = tanh_range(param, *cfg.cont_range, initial=None)   # 0.8
        
    luminance = rgb2lum(img)
    zero = torch.zeros_like(luminance)
    one = torch.ones_like(luminance)

    luminance = torch.where(luminance < 0, zero, luminance)
    luminance = torch.where(luminance > 1, one, luminance)

    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    
    return lerp(img, contrast_image, param)


class GammaFilter(nn.Module): 

  def __init__(self, inputChannel):
    super(GammaFilter, self).__init__()
    
    self.num_filter_parameters = 1
    self.channels = inputChannel

  def forward(self, img, param):
    param = torch.exp(tanh_range(param, -cfg.gamma_range, cfg.gamma_range, initial=None))   

    zero = torch.zeros_like(img) + 0.00001
    img = torch.where(img <= 0, zero, img)

    return torch.pow(img, param)


class DefogFilter(nn.Module):
  """
  Dark channel haze remove, Author: Zhipan Wang, Email:1044625113@qq.com
  """
  def __init__(self, inputChannel):
    super(DefogFilter, self).__init__()
    
    self.num_filter_parameters = 1

  def forward(self, img, param):
    param = tanh_range(param, *cfg.defog_range, initial=None)

    batch, channel, h, w = img.shape
    newImg = img.clone()
    
    for i in range(0, batch):
      param_singleImg = param[i, :, :, :].clone()   
      param_singleImg = torch.squeeze(param_singleImg)

      singleImg = img[i, :, :, :]

      singleImg = hazeRemove(singleImg, w=param_singleImg)       

      singleImg = singleImg.permute(2, 0, 1)
      newImg[i, :, :, :] = singleImg

          
    return newImg
      