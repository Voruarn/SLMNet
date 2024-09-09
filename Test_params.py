import os
import torch
from torch.autograd import Variable

from network.SLMNet import SLMNet

from thop import profile



if __name__ == "__main__":
    print('Test Model parameters !')
    model=SLMNet()
    model.eval()
    batch_size=1
    input=torch.rand(batch_size,3,256,256)
    flops, params = profile(model, inputs=(input, ))

    GFLOPs=10**9
    Million=10**6
    print('FLOPs:{:.2f}G'.format((flops/GFLOPs)/batch_size), end=', ')

    print('params:{:.2f}M'.format(params/Million))

 



"""
SLMNet: FLOPs:9.34G, params:30.70M

"""



