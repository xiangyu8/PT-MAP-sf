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
from data.datamgr import SimpleDataManager , SimpleDataManager_both,SetDataManager
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

def extract_feature(val_loader_dct, model_dct,val_loader_plain, model_plain, checkpoint_dir, tag='last'):
    save_dir = '{}/{}'.format(checkpoint_dir, tag)
    max_count = len(val_loader_dct)*val_loader_dct.batch_size 
    if os.path.isfile(save_dir + '/output_both.plk'):
        data = load_pickle(save_dir + '/output_both.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    print("what?")
    #model.eval()
    with torch.no_grad():
        
        output_dict = collections.defaultdict(list)
        print("ing...")
        all_feats = np.zeros([len(val_loader_dct),1280])
        count = 0
        for i, (inputs, labels) in enumerate(val_loader_dct):
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs,_ = model_dct(inputs)
            outputs = outputs.cpu().data.numpy()
           # print("inside extracter:", outputs.shape)            
            for out, label in zip(outputs, labels):
             #   print("out: ", out.shape, type(out))
    #            output_dict[label.item()].append(out)
                #if all_feats is None:
                #    all_feats = f.create_dataset('all_feats', [max_count] + [1280] , dtype='f')
                out_tmp = out.reshape(1,640)
                all_feats[count:count+1, 0:640] = out_tmp
            count = count + 1                

        count = 0
        for i, (inputs, labels) in enumerate(val_loader_plain):
            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs,_ = model_plain(inputs)
            outputs = outputs.cpu().data.numpy()
            #print("inside extracter:", outputs.shape)            
            for out, label in zip(outputs, labels):
               # output_dict[label.item()].append(out)
                #if all_feats is None:
                #    all_feats = f.create_dataset('all_feats', [max_count] + [1280] , dtype='f')
                out_tmp = out.reshape(1,640)
                all_feats[count:count+1, 640:] = out_tmp
                out_final = all_feats[count]
                output_dict[label.item()].append(out_final)
        #        print("out: ", out_tmp.shape, out_final.shape)
        #        print(out_final)
            count = count + 1
        print(all_feats.shape)
        all_info = output_dict
        save_pickle(save_dir + '/output_both.plk', all_info)
        return all_info

if __name__ == '__main__':
    params = parse_args('test')

    loadfile = configs.data_dir[params.dataset] + 'novel.json'

    if params.dct_status:
        image_size = 56
    else:
        image_size = 84
    if params.dataset == 'cifar':
        image_size = 32
        params.num_classes = 64
        
    
    #if params.dataset == 'miniImagenet' or params.dataset == 'CUB':
    datamgr       = SimpleDataManager_both(image_size, batch_size = 1)
    novel_loader_dct      = datamgr.get_data_loader_dct(loadfile, aug = False)
    novel_loader_plain      = datamgr.get_data_loader(loadfile, aug = False)

    checkpoint_dir_plain = '%s/checkpoints/%s/%s_%s_%sway_%sshot_aug_pretrained' %(configs.save_dir, params.dataset, params.model, params.method,params.test_n_way, params.n_shot)
    checkpoint_dir_dct = '%s/checkpoints/%s/%s_%s_%sway_%sshot_aug_dct' %(configs.save_dir, params.dataset, params.model, params.method,params.test_n_way, params.n_shot)

    modelfile_plain   = get_best_file(checkpoint_dir_plain)
    modelfile_dct = get_best_file(checkpoint_dir_dct)
    print(checkpoint_dir_plain, checkpoint_dir_dct)

    if params.model == 'WideResNet28_10':
        model_plain = wrn_mixup_model.wrn28_10(num_classes=params.num_classes, dct_status = False)
        model_dct = wrn_mixup_model.wrn28_10(num_classes=params.num_classes, dct_status = True)
    elif params.model == 'ResNet18':
	    model = res_mixup_model.resnet18(num_classes=params.num_classes)

    model_plain = model_plain.cuda()
    model_dct = model_dct.cuda()
    cudnn.benchmark = True

    checkpoint_plain = torch.load(modelfile_plain)
    checkpoint_dct = torch.load(modelfile_dct)
    state_plain = checkpoint_plain['state']
    state_dct = checkpoint_dct['state']
    state_keys_plain = list(state_plain.keys())
    state_keys_dct = list(state_dct.keys())

    callwrap = False
    if 'module' in state_keys_plain[0]:
        callwrap = True
    if callwrap:
        model_plain = WrappedModel(model_plain)
    model_dict_load_plain = model_plain.state_dict()
    model_dict_load_plain.update(state_plain)
    model_plain.load_state_dict(model_dict_load_plain)
    model_plain.eval()
    callwrap = False
    if 'module' in state_keys_dct[0]:
        callwrap = True
    if callwrap:
        model_dct = WrappedModel(model_dct)
    model_dict_load_dct = model_dct.state_dict()
    model_dict_load_dct.update(state_dct)
    model_dct.load_state_dict(model_dict_load_dct)
    model_dct.eval()

    output_dict=extract_feature(novel_loader_dct, model_dct,novel_loader_plain, model_plain, checkpoint_dir_plain, tag='last') 
    print("features saved!")
