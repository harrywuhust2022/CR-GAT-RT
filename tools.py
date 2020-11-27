import torch
import torch.nn.functional as F
from utils.helpers import colorize_mask
import cv2
import os
import matplotlib.pyplot as plt
def calculate_correct_map(output, target, num_class):
    _, predict = torch.max(output.data, 1)
    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)
    correct = ((predict == target) * labeled)
    return correct


def getw(output):
    out = F.softmax(output, dim=1).permute(0, 2, 3, 1)
    values, indices = torch.topk(out, 2)
    vl = values.permute(3, 0, 1, 2)
    vl = torch.clamp(vl, 1e-8, 0.999998)

    from scipy.stats import norm
    surrogate = vl[0] - vl[1]
    vl = norm.ppf(vl[0].detach().cpu()) - norm.ppf(vl[1].detach().cpu())
    vl = torch.from_numpy(vl).float()

    return torch.cat((vl.unsqueeze(1), vl.unsqueeze(1), vl.unsqueeze(1)), dim=1), vl, torch.cat(
        (surrogate.unsqueeze(1), surrogate.unsqueeze(1), surrogate.unsqueeze(1)), dim=1)


def get_CR(output):
    out = F.softmax(output, dim=1).permute(0, 2, 3, 1)
    values, indices = torch.topk(out, 2)
    vl = values.permute(3, 0, 1, 2)
    surrogate = vl[0] - vl[1]
    return surrogate


'''
def getw2(output):
    out = F.softmax(output,dim=1).permute(0,2,3,1)
    values, indices = torch.topk(out, 2)
    vl = values.permute(3, 0, 1,2)
    from scipy.stats import norm
    surrogate=vl[0]-vl[1]
    vl=norm.ppf(vl[0].detach().cpu())-norm.ppf(vl[1].detach().cpu())
    vl=torch.from_numpy(vl).float()
    weight=torch.ones((5,5))
    weight=weight/25
    weight=weight.unsqueeze(0).unsqueeze(0)
    vl=F.conv2d(vl.unsqueeze(1),weight,padding=2)
    return torch.cat((vl,vl,vl),dim=1),vl.squeeze(1),surrogate
'''