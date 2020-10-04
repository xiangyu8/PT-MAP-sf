import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import data.feature_loader as feat_loader
from sklearn.manifold import TSNE

sns.set_context("notebook", font_scale = 1.1)
sns.set_style("ticks")
len_test = 20
len_type = 5
novel_file = "./novel.hdf5"
novel_both_file = "./novel_both.hdf5"

features = feat_loader.init_loader(novel_file)
features_both = feat_loader.init_loader(novel_both_file)

class_list = features.keys()
select_class = random.sample(class_list, len_type)

print(class_list)
#select_class = [84,85,86,87,88,89, 90, 91]
print(select_class)
z_all = []
z_both_all = []
for cl in select_class:
	img_feat = features[cl]
	img_both_feat = features_both[cl]

	print('len of cases: ',len(img_feat))
	perm_ids = np.random.permutation(len(img_feat)).tolist()
	z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(len_test) ] )
	z_both_all.append( [ np.squeeze( img_both_feat[perm_ids[i]]) for i in range(len_test) ] )
 
#z_all = torch.from_numpy(np.array(z_all) )   
z_all = np.array(z_all)
z_both_all = np.array(z_both_all)
print(type(z_all))
print(z_all.shape)
print(z_both_all.shape)

z_all = z_all.reshape(-1,z_all.shape[2])
print(z_all.shape)
print(z_both_all.shape)
tsne = TSNE(n_components = 2).fit_transform(z_all)
print(tsne.shape)
z_both_all = z_both_all.reshape(-1,z_both_all.shape[2])
tsne_both = TSNE(n_components = 2).fit_transform(z_both_all)
print(tsne_both.shape)

def scale_to_01_range(x):

	value_range = (np.max(x) - np.min(x))
	starts_from_zero = x - np.min(x)
	return starts_from_zero / value_range

tx = tsne[:, 0]
ty = tsne[:, 1]

#tx = scale_to_01_range(tx)
#ty = scale_to_01_range(ty)
print(tx.shape)

tx_both = tsne_both[:, 0]
ty_both = tsne_both[:, 1]

#tx_both = scale_to_01_range(tx_both)
#ty_both = scale_to_01_range(ty_both)


# nitialize a matplotlib plot
fig = plt.figure(1)
ax = fig.add_subplot(111)

labels = {}
for i in range(len_type):
	labels_sub = []
	for j in range(len_test):
		labels_sub.append(i*len_test + j)
	labels[str(i)] = labels_sub

for label in labels:
	indices = labels[label]
		
	current_tx = np.take(tx, indices)
	current_ty = np.take(ty, indices)


	ax.scatter(current_tx, current_ty, label=label)

fig = plt.figure(2)
ax = fig.add_subplot(111)

labels = {}
for i in range(len_type):
	labels_sub = []
	for j in range(len_test):
		labels_sub.append(i*len_test + j)
	labels[str(i)] = labels_sub

for label in labels:
	indices = labels[label]
		
	current_tx_both = np.take(tx_both, indices)
	current_ty_both = np.take(ty_both, indices)

	ax.scatter(current_tx_both, current_ty_both, label=label)


plt.show()

