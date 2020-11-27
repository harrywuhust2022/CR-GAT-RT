import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import torch.nn.functional as F
import math
import numpy as np
import torchvision
from utils import losses
from utils.torchsummary import summary
from utils.metrics import eval_metrics
from utils.helpers import colorize_mask

from defense import zoom_defense
import time
from attack import get_adv_examples_FGSM
from attack import get_adv_examples_L2
from attack import get_adv_examples_LINF
from attack import get_adv_examples_S
from attack import get_adv_examples
from attack import getw
from statistic import visiualize
from statistic import visiualizelst
def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class METRICS:
    def __init__(self, num_classes):
        self.total_correct = 0
        self.total_label = 0
        self.total_inter = 0
        self.total_union = 0
        self.num_classes = num_classes

    def update_seg_metrics(self, correct, labeled, inter, union):

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = []
        for i in range(0, self.num_classes):
            if (self.total_union[i] > 0):
                IoU.append(self.total_inter[i] / self.total_union[i])
        mIoU = sum(IoU) / len(IoU)
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

myloss=0


def main(args,config, resume):
    # DATA LOADERS
    global myloss, bordermark
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    num_classes = train_loader.dataset.num_classes
    palette = train_loader.dataset.palette


    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])

    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
    checkpoint = torch.load(args.binname,map_location=torch.device('cpu') )
    checkpoint=checkpoint["state_dict"]

    model.set_mean_std(train_loader.MEAN,train_loader.STD)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    cnt = 0
    myloss=getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    atk_m1 = METRICS(num_classes)
    atk_m2 = METRICS(num_classes)
    atk_m3 = METRICS(num_classes)
    atk_m4 = METRICS(num_classes)


    for batch_idx, (data, targ) in enumerate(val_loader):
        cnt+=1
        
        torch.cuda.empty_cache()
        data = data.float()
        data = data.requires_grad_()
        targ=targ.cuda()

        adv = get_adv_examples_S(data, targ, model, myloss,25,10,15,palette)
        output =model(torch.clamp(adv+0.04*torch.randn(data.shape),0,1))['out']
        seg_metrics = eval_metrics(output, targ.cuda(), num_classes) 
        atk_m1.update_seg_metrics(*seg_metrics)
        pixAcc1, mIoU1, _ = atk_m1.get_seg_metrics().values()


        adv = get_adv_examples_S(data, targ, model, myloss,50,10,15,palette)
        output =model(torch.clamp(adv+0.04*torch.randn(data.shape),0,1))['out'] 
        seg_metrics = eval_metrics(output, targ.cuda(), num_classes)
        atk_m2.update_seg_metrics(*seg_metrics)
        pixAcc2, mIoU2, _ = atk_m2.get_seg_metrics().values()

        adv = get_adv_examples_S(data, targ, model, myloss,100,10,15,palette) 
        output =model(torch.clamp(adv+0.04*torch.randn(data.shape),0,1))['out'] 
        seg_metrics = eval_metrics(output, targ.cuda(), num_classes) 
        atk_m3.update_seg_metrics(*seg_metrics)
        pixAcc3, mIoU3, _ = atk_m3.get_seg_metrics().values()

        adv = get_adv_examples_S(data, targ, model, myloss,150,10,20,palette) 
        output =model(torch.clamp(adv+0.04*torch.randn(data.shape),0,1))['out'] 
        seg_metrics = eval_metrics(output, targ.cuda(), num_classes) 
        atk_m4.update_seg_metrics(*seg_metrics)
        pixAcc4, mIoU4, _ = atk_m4.get_seg_metrics().values()

        with open('./results.txt', 'a') as f:
            print("ROUND %d"%(cnt),file=f)
            print(" OUR 25:  %f %f"%(pixAcc1, mIoU1),file=f)
            print(" OUR 50:  %f %f"%(pixAcc2, mIoU2),file=f)
            print(" OUR 100:  %f %f"%(pixAcc3, mIoU3),file=f)
            print(" OUR 150:  %f %f"%(pixAcc4, mIoU4),file=f)
            f.close()

if __name__ == '__main__':

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configvoc_psp.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-b', '--binname', default='', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    print(args.binname)
    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args,config, args.resume)

