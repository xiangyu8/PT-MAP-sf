# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import cv2
import json
import numpy as np
import torchvision.transforms as transforms
import os
# Optimized for DCT
# Upsampling in the compressed domain
import sys
import random
from datasets.vision import VisionDataset
import cv2
import os.path
from turbojpeg import TurboJPEG
from datasets import train_y_mean_resized, train_y_std_resized, train_cb_mean_resized, train_cb_std_resized, \
    train_cr_mean_resized, train_cr_std_resized
from jpeg2dct.numpy import loads

identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            print(cl)
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
        print('end of SetDataSet for loop')
    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SetDataset_dct:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = ImageFolderDCT_dct(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]



#########################
#########################below was revised by Xiangyu

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def opencv_loader(path, colorSpace='YCrCb'):
    image = cv2.imread(str(path))
    print('opencv_loader: ',path)
  #  cv2.imwrite('.datasets/cvtransforms/test/raw.jpg', image)
    print('size of image: ',image.shape)
    if colorSpace == "YCrCb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('/mnt/ssd/kai.x/work/code/iftc/datasets/cvtransforms/test/ycbcr.jpg', image)
    elif colorSpace == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def default_loader(path, backend='opencv', colorSpace='YCrCb'):
    from torchvision import get_image_backend
    if backend == 'opencv':
        return opencv_loader(path, colorSpace=colorSpace)
    elif get_image_backend() == 'accimage' and backend == 'acc':
        return accimage_loader(path)
    elif backend == 'pil':
        return pil_loader(path)
    else:
        raise NotImplementedError

def adjust_size(y_size, cbcr_size):
    if y_size == cbcr_size:
        return y_size, cbcr_size
    elif np.mod(y_size, 2) == 1:
        y_size -= 1
        cbcr_size = y_size // 2
    return y_size, cbcr_size

class DatasetFolderDCT(VisionDataset):
    def __init__(self, data_file,loader,  transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, i):
        path = os.path.join(self.meta['image_names'][i])
        backend = 'dct'
        if backend == 'opencv':
            sample = self.loader(path, backend = 'opencv', colorSpace='BGR')

        elif backend == 'dct':
            try:
                with open(path, 'rb') as src:
                    buffer = src.read()
                dct_y, dct_cb, dct_cr = loads(buffer)
            except:
                notValid = True
                while notValid:
                    i = random.randint(0, len(self.samples)-1)
                    path = os.path.join(self.sub_meta[i])
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        notValid = True

            if len(dct_y.shape) != 3:
                notValid = True
                while notValid:
                    i = random.randint(0, len(self.sub_meta)-1)
                    path = os.path.join(self.sub_meta[i])
                    print(path)
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        print(path)
                        notValid = True

            y_size_h, y_size_w = dct_y.shape[:-1]
            cbcr_size_h, cbcr_size_w = dct_cb.shape[:-1]

            y_size_h, cbcr_size_h = adjust_size(y_size_h, cbcr_size_h)
            y_size_w, cbcr_size_w = adjust_size(y_size_w, cbcr_size_w)
            dct_y = dct_y[:y_size_h, :y_size_w]
            dct_cb = dct_cb[:cbcr_size_h, :cbcr_size_w]
            dct_cr = dct_cr[:cbcr_size_h, :cbcr_size_w]
            sample = [dct_y, dct_cb, dct_cr]
    
     #       print('shape of dct_y: ',dct_cb.shape)    

            y_h, y_w, _ = dct_y.shape
            cbcr_h, cbcr_w, _ = dct_cb.shape

        if self.transform is not None:
           # dct_y, dct_cb, dct_cr = self.transform(sample)
            dct_y = self.transform(dct_y)
            dct_cb = self.transform(dct_cb)
            dct_cr = self.transform(dct_cr)

        print('target: ',self.meta['image_labels'][i])

       # target = self.target_transform(self.meta['image_labels'][i])
        target = self.meta['image_labels'][i]

        if dct_cb is not None:
            image = torch.cat((dct_y, dct_cb, dct_cr), dim=1)
            return image, target
        else:
            return dct_y, target

    def __len__(self):
        return len(self.meta['image_names'])

class DatasetFolderDCT_dct(VisionDataset):
    def __init__(self,  sub_meta,loader, cl,transform, target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
      

    def __getitem__(self, i):
        path = os.path.join(self.sub_meta[i])
        # path = '/storage-t1/user/kaixu/datasets/ILSVRC2012/test/train/n02447366/n02447366_23489.JPEG'
        print('path of DCT_dct',path)
        backend = 'dct'
        if backend == 'opencv':
            sample = self.loader(path, backend='opencv', colorSpace='BGR')
        elif backend == 'dct':
            try:
                with open(path, 'rb') as src:
                    buffer = src.read()
                dct_y, dct_cb, dct_cr = loads(buffer)
            except:
                notValid = True
                while notValid:
                    i = random.randint(0, len(self.samples)-1)
                    path = os.path.join(self.sub_meta[i])
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        notValid = True

            if len(dct_y.shape) != 3:
                notValid = True
                while notValid:
                    i = random.randint(0, len(self.sub_meta)-1)
                    path = os.path.join(self.sub_meta[i])
                    print(path)
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        print(path)
                        notValid = True

            y_size_h, y_size_w = dct_y.shape[:-1]
            cbcr_size_h, cbcr_size_w = dct_cb.shape[:-1]

            y_size_h, cbcr_size_h = adjust_size(y_size_h, cbcr_size_h)
            y_size_w, cbcr_size_w = adjust_size(y_size_w, cbcr_size_w)
            dct_y = dct_y[:y_size_h, :y_size_w]
            dct_cb = dct_cb[:cbcr_size_h, :cbcr_size_w]
            dct_cr = dct_cr[:cbcr_size_h, :cbcr_size_w]
            sample = [dct_y, dct_cb, dct_cr]

            y_h, y_w, _ = dct_y.shape
            cbcr_h, cbcr_w, _ = dct_cb.shape

####################

        if self.transform is not None:
            dct_y, dct_cb, dct_cr = self.transform(sample)

        #print('self.cl: ',self.cl)

        target = self.target_transform(self.cl)

        if dct_cb is not None:
            image = torch.cat((dct_y, dct_cb, dct_cr), dim=1)
            return image, target
        else:
            return dct_y, target


    def __len__(self):
        return len(self.sub_meta)


class ImageFolderDCT(DatasetFolderDCT):
    def __init__(self, root,transform=None, target_transform=None,loader=default_loader):
        super(ImageFolderDCT, self).__init__(root,loader,transform=transform,target_transform=target_transform)

class ImageFolderDCT_dct(DatasetFolderDCT_dct):
    def __init__(self, sub_meta,cl, transform=None, target_transform=None,loader=default_loader):
        super(ImageFolderDCT_dct, self).__init__(sub_meta, loader,cl, transform=transform,target_transform=target_transform)

