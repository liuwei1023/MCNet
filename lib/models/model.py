from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Logger import system_log
from config import system_config


import torchvision.models as models
import torch
import torch.nn as nn
import os
import time 

from .networks.MCNet import get_MobileNet_CenterNet_model as get_MCModel

_model_factory = {
    "mbv3_CenterNet": get_MCModel
}

def create_model(model_name):
    start = time.time()
    get_model =_model_factory[model_name]
    model = get_model(system_config.Detector["num_class"])
    end = time.time()

    system_log.WriteLine(f"create model {model_name}, cost time: {(end-start):.8f}sec!")
    return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {
        "epoch": epoch,
        "state_dict": state_dict
    }

    if not (optimizer is None):
        data['optimizer'] = optimizer

    torch.save(data, path)
    system_log.WriteLine(f"save model to {path}")


def load_model(model, model_path, optimizer=None, lr=None, device='gpu'):
    start = time.time()

    start_epoch = 0
    
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()

    system_log.WriteLine(f"loaded {model_path}, epoch {checkpoint['epoch']}, device: {device}")

    if optimizer is not None and lr is not None:
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr 
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            system_log.WriteLine(f"resumed optimizer with start lr: {start_lr}")

        else:
            system_log.WriteLine(f"No optimizer parameters in checkpoint.")

    end = time.time()
    system_log.WriteLine(f"load model done. cost time: {(end-start):.8f}sec!")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model 


    




