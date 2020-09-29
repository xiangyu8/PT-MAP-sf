import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re
import shutil
from tqdm import tqdm as tqdm

import os


cwd = os.getcwd()
cwd = os.getcwd()
datadir = cwd.split('filelists')[0]

print(datadir)
data_path = './miniImagenet/Images/'
print(data_path)
savedir = './'
dataset_list = ['base', 'val', 'novel']


cl = -1
folderlist = []

datasetmap = {'base':'train','val':'val','novel':'test'};
filelists = {'base':{},'val':{},'novel':{} }
filelists_flat = {'base':[],'val':[],'novel':[] }
labellists_flat = {'base':[],'val':[],'novel':[] }

# Find class identities
classes = []
for root, _, files in os.walk(data_path):
    for f in files:
        #if f.endswith('.jpg'):
        classes.append(f[:-12])

classes = list(set(classes))

classes_arr = classes[:]
#data_path_new = './image_saliency/Image_MB/'

data_path_new = './miniImagenet/Images/'
#print(len(classes))
#shutil.rmtree(data_path_new)
"""
for c in classes_arr:
    print(data_path_new +f'{c}')
    os.mkdir(data_path_new +f'{c}')

# Move images to correct location
for root, _, files in os.walk(data_path):
    for f in tqdm(files, total=600*100):
        if f.endswith('.jpg'):
            class_name = f[:-12]
            image_name = f[-12:]
            src = f'{root}/{f}'
            dst = data_path_new + f'{class_name}/{image_name}'
            shutil.copy(src, dst)
"""
for dataset in dataset_list:
   # mark = 0
    with open(datasetmap[dataset] + ".csv", "r") as lines:
        for i, line in enumerate(lines):
            if i==0:
                print("i = 0\n")
                continue
            fid, _ , label = re.split(',|\.', line)
            label = label.replace('\n','')
            if not label in filelists[dataset]:
                folderlist.append(label)
                filelists[dataset][label] = []
                fnames = listdir( join(data_path_new, label) )
              #  for fname in fnames:
              #      print(re.split('_|\.', fname)[0])
                fname_number = [ int(re.split('_|\.', fname)[0]) for fname in fnames]
                sorted_fnames = list(zip( *sorted(  zip(fnames, fname_number), key = lambda f_tuple: f_tuple[1] )))[0]

            fid = int(fid[-5:])-1
            fname = join( data_path_new,label, sorted_fnames[i%600-1] )
            filelists[dataset][label].append(fname)
            mark = i
  #  print(mark)

    for key, filelist in filelists[dataset].items():
        cl += 1
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist()

for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
