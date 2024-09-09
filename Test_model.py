import torch
import torch.nn.functional as F

import numpy as np
from network.SLMNet import SLMNet
from SSMAE import ssmae

if __name__=='__main__':
    print('Test Model')

    input=torch.rand(2,3,256,256).cuda()
    # mask=torch.rand(2,1,256,256).cuda()

    model = SLMNet().cuda()
    model.eval()
    output=model(input)

    # model = ssmae().cuda()
    # model.eval()
    # output=model(input, mask)

    for x in output:
        print('x.shape:',x.shape)

