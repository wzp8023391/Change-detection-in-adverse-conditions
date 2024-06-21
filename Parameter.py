# coding:utf-8
import torch
from thop import profile
import time

from module.model import IA_CDNet 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    inputChannel, numClass = 6, 2
    
    input = torch.randn(6, 6, 512, 512).to(device)
    
    model = IA_CDNet(num_classes=numClass, in_channels=inputChannel).to(device) 
    
    
    flops, params = profile(model, (input,))  # Parameter
    print('diff_flops: G', flops/1e9, 'diff_params: M', params/1e6)
    
    time_s = time.time()                      # inference time
    result = model(input)
    
    time_e = time.time()
    time_all = time_e - time_s
    print("cost time:", time_all)
    
