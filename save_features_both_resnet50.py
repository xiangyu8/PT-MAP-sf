import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager, SimpleDataManager_both
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 
import wrn_mixup_model
import torch.nn as nn
#from methods.resnet import ResNetDCT_Upscaled_Static


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)


def save_features(model, data_loader, outfile ,params ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        print("inside save_features_both: ", len(data_loader))
        print(x.shape,y.shape)
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))

        if torch.cuda.is_available():
            x = x.cuda()
        x_var = Variable(x)
        if params.method == 'manifold_mixup' or params.method == 'S2M2_R':
            feats,_ = model(x_var)
        else:
            feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


def save_features_both(model_dct, model_plain, data_loader_dct, data_loader_plain, outfile ,params ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader_dct)*data_loader_dct.batch_size
    all_labels  = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader_dct):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader_dct)))
        if torch.cuda.is_available():
            x = x.cuda()
        x_var = Variable(x)
        if params.method == 'manifold_mixup' or params.method == 'S2M2_R':
            feats_dct,_ = model_dct(x_var)
        else:
            feats_dct = model_dct(x_var)
        
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + [2048*2] , dtype='f')
        all_feats[count:count+feats_dct.size(0),0:2048] = feats_dct.data.cpu().numpy()
        all_labels[count:count+feats_dct.size(0)] = y.cpu().numpy()
        count = count + feats_dct.size(0)

    count=0
    for i, (x,y) in enumerate(data_loader_plain):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader_plain)))
        if torch.cuda.is_available():
            x = x.cuda()
        x_var = Variable(x)
        if params.method == 'manifold_mixup' or params.method == 'S2M2_R':
            feats_plain,_ = model_plain(x_var)
        else:
            feats_plain = model_plain(x_var)
        
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats_plain.size()[1:]+feats_plain.size()[1:]) , dtype='f')
        all_feats[count:count+feats_dct.size(0),2048:]= feats_plain.data.cpu().numpy() 
        """
        if all_labels[count:count+feats_dct.size(0)] != y.cpu().numpy:
            print("Error from different label")
            print(all_labels[count:count+feats_dct.size(0)]!=y.cpu().numpy())
            all_labels[count:count+feats_dct.size(0)] = y.cpu().numpy()
        """
        count = count + feats_plain.size(0)
    print(all_feats.shape)
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    params = parse_args('save_features')
    
    if params.dataset == 'cifar':
        image_size = 32
    elif params.method =='dct':
        image_size = 448
    else:
        image_size = 224

    split = params.split
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    checkpoint_dir_plain = '%s/checkpoints/%s/%s_plain' %(configs.save_dir, params.dataset, params.model)
    checkpoint_dir_dct = '%s/checkpoints/%s/%sdct_dct' %(configs.save_dir, params.dataset, params.model)

    print("chechpoint_dir_plain: ", checkpoint_dir_plain)
    print("checkpoint_dir_dct:", checkpoint_dir_dct)

    if params.save_iter != -1:
        modelfile_plain   = get_assigned_file(checkpoint_dir_plain,params.save_iter)
        modelfile_dct   = get_assigned_file(checkpoint_dir_dct,params.save_iter)
    else:
        modelfile_plain   = get_best_file(checkpoint_dir_plain)
        modelfile_dct   = get_best_file(checkpoint_dir_dct)    


    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir_plain.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ "_both.hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir_plain.replace("checkpoints","features"), split + "_both.hdf5") 

    datamgr         = SimpleDataManager_both(image_size, batch_size = 1)

    data_loader_dct      = datamgr.get_data_loader_dct(loadfile, aug = False)    
    data_loader_plain      = datamgr.get_data_loader(loadfile, aug = False)

    if params.method == 'manifold_mixup':
        if params.dataset == 'cifar':
            model = wrn_mixup_model.wrn28_10(64)
        else:
            model = wrn_mixup_model.wrn28_10(200)
    elif params.method == 'S2M2_R':
        if params.dataset == 'cifar':
            model = wrn_mixup_model.wrn28_10(64 , loss_type = 'softmax')
        else:
            model = wrn_mixup_model.wrn28_10(200)
    else:
        model_plain = model_dict[params.model]()
        model_dct = model_dict[params.model + 'dct']()

   # print(checkpoint_dir , modelfile)
    if params.method == 'manifold_mixup' or params.method == 'S2M2_R' :
        if torch.cuda.is_available():
            model = model.cuda()
        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        callwrap = False
        if 'module' in state_keys[0]:
            callwrap = True

        if callwrap:
            model = WrappedModel(model) 

        model_dict_load = model.state_dict()
        model_dict_load.update(state)
        model.load_state_dict(model_dict_load)
    
    else:
        if torch.cuda.is_available():
            model_dct = model_dct.cuda()
            model_plain = model_plain.cuda()
        tmp_plain = torch.load(modelfile_plain)
        tmp_dct = torch.load(modelfile_dct)
        state_plain = tmp_plain['state']
        state_dct = tmp_dct['state']
        callwrap = False
        state_keys_plain = list(state_plain.keys())
        state_keys_dct  = list(state_dct.keys())

        for i, key in enumerate(state_keys_plain):
            if 'module' in key and callwrap == False:
                callwrap = True
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state_plain[newkey] = state_plain.pop(key)
            else:
                state_plain.pop(key)

        for i, key in enumerate(state_keys_dct):
            if 'module' in key and callwrap == False:
                callwrap = True
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state_dct[newkey] = state_dct.pop(key)
            else:
                state_dct.pop(key)                
    
        if callwrap:
            model_plain = WrappedModel(model_plain) 
            model_dct   = WrappedModel(model_dct)

        model_plain.load_state_dict(state_plain)   
        model_dct.load_state_dict(state_dct)
   
    model_plain.eval()
    model_dct.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features_both(model_dct, model_plain, data_loader_dct, data_loader_plain, outfile , params)
