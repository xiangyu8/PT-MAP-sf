from __future__ import print_function

import argparse
import csv
import os
import collections
import pickle
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from io_utils import parse_args
from data.datamgr import SimpleDataManager , SetDataManager
import configs

import wrn_mixup_model
import res_mixup_model

import torch.nn.functional as F

from io_utils import parse_args, get_resume_file ,get_assigned_file, get_best_file
from os import path

use_gpu = torch.cuda.is_available()

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 
    def forward(self, x):
        return self.module(x)

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def extract_feature(val_loader, model, checkpoint_dir, tag='last'):
    save_dir = '{}/{}'.format(checkpoint_dir, tag)
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    #model.eval()
    with torch.no_grad():
        
        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs,_ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)
    
        all_info = output_dict
        save_pickle(save_dir + '/output.plk', all_info)
        return all_info

if __name__ == '__main__':
    params = parse_args('test')

    loadfile = configs.data_dir[params.dataset] + 'novel.json'

    if params.dct_status:
        image_size = 56
    else:
        if params.dataset =='cifar':
            image_size = 32
            params.num_classes = 64
        else:
            image_size = 84
        
    #if params.dataset == 'miniImagenet' or params.dataset == 'CUB':
    datamgr       = SimpleDataManager(image_size, batch_size = 16)
    if params.dct_status:
        novel_loader      = datamgr.get_data_loader_dct(loadfile, aug = False)
  
    else:
        novel_loader      = datamgr.get_data_loader(loadfile, aug = False)
        params.channels = 3

    checkpoint_dir = '%s/checkpoints/%s/%s_%s_%sway_%sshot' %(configs.save_dir, params.dataset, params.model, params.method,params.test_n_way, params.n_shot)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if params.dct_status:
        checkpoint_dir += '_dct'

    modelfile   = get_best_file(checkpoint_dir)
    print(checkpoint_dir)

    if params.model == 'WideResNet28_10':
        model = wrn_mixup_model.wrn28_10(num_classes=params.num_classes, dct_status = params.dct_status)
    elif params.model == 'ResNet18':
	    model = res_mixup_model.resnet18(num_classes=params.num_classes)

    model = model.cuda()
    cudnn.benchmark = True

    checkpoint = torch.load(modelfile)
    state = checkpoint['state']
    state_keys = list(state.keys())

    callwrap = False
    if 'module' in state_keys[0]:
        callwrap = True
    if callwrap:
        model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    model.eval()
    output_dict=extract_feature(novel_loader, model, checkpoint_dir, tag='last') 
    print("features saved!")
