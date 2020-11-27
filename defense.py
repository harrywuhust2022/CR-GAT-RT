import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def zoom_defense(model,x):        #get output
    h, w = x.size()[2], x.size()[3]
    newh=random.randint(int(h*0.75), int(h*1.25))
    neww = random.randint(int(w * 0.75), int(w * 1.25))
    x = F.interpolate(x, size=(newh, neww), mode='bilinear', align_corners=True)
    output=model(x)['out']
    output=F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
    return output

def random_inference(model,x):
    output = model(torch.clamp(x+0.04*torch.randn(x.shape),0,1))['out']
    return output