# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import datasets.cvtransforms as transforms_dct
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset,EpisodicBatchSampler
from abc import abstractmethod
from datasets import train_upscaled_static_mean, train_upscaled_static_std
from datasets import train_y_mean_resized, train_y_std_resized, train_cb_mean_resized, train_cb_std_resized, train_cr_mean_resized, train_cr_std_resized

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug: # deleted 'ImageJitter' by Xiangyu 06/23/2020 to compare with dct
            transform_list = ['RandomResizedCrop', 'ImageJitter','RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
        
        input_size1 = 512
        input_size2 = 448

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_composed_transform_dct(self, aug = False):
        input_size1 = 512
        input_size2 = 448
        if aug==False:
            transform = transforms_dct.Compose([ #transform_funcs,
               # transforms_dct.Resize(int(self.image_size*1.15)),
              #  transforms_dct.CenterCrop(self.image_size),
              #  transforms_dct.Resize(448),
                transforms_dct.Resize(int(448*1.15)),
                transforms_dct.CenterCrop(448),
                transforms_dct.GetDCT(),
                transforms_dct.UpScaleDCT(size=56),
                transforms_dct.ToTensorDCT(),
                transforms_dct.SubsetDCT(channels=24),
                transforms_dct.Aggregate(),
                transforms_dct.NormalizeDCT(
                      #  train_y_mean_resized,  train_y_std_resized,
                      #  train_cb_mean_resized, train_cb_std_resized,
                      #  train_cr_mean_resized, train_cr_std_resized),
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                    channels=24
                )
                #transforms_dct.Aggregate()
            ])
        else:
            transform = transforms_dct.Compose([ #transform_funcs,
             #   transforms_dct.RandomResizedCrop(self.image_size),
             #   transforms_dct.Resize(448),
                transforms_dct.RandomResizedCrop(448),
                transforms_dct.ImageJitter(self.jitter_param),
                transforms_dct.RandomHorizontalFlip(),
                transforms_dct.GetDCT(),
                transforms_dct.UpScaleDCT(size=56),
                transforms_dct.ToTensorDCT(),
                transforms_dct.SubsetDCT(channels=24),
                transforms_dct.Aggregate(),
                transforms_dct.NormalizeDCT(
                      #  train_y_mean_resized,  train_y_std_resized,
                      #  train_cb_mean_resized, train_cb_std_resized,
                      #  train_cr_mean_resized, train_cr_std_resized),
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                    channels=24
                )
             ])
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform, dct_status = False)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 8, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

    def get_data_loader_dct(self, data_file, aug):
        transform = self.trans_loader.get_composed_transform_dct(aug)
        dataset = SimpleDataset(data_file, transform,dct_status =True)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 16, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)  
        return data_loader 

class SimpleDataManager_both(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager_both, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform, dct_status = False)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = False, num_workers = 8, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

    def get_data_loader_dct(self, data_file, aug):
        transform = self.trans_loader.get_composed_transform_dct(aug)
        dataset = SimpleDataset(data_file, transform,dct_status =True)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = False, num_workers = 16, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)  
        return data_loader 

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform, dct_status=False )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 8, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

    def get_data_loader_dct(self, data_file, aug): 
        transform = self.trans_loader.get_composed_transform_dct(aug)
        dataset = SetDataset( data_file , self.batch_size, transform, dct_status=True ) 
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 16, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


