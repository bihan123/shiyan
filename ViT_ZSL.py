#!/usr/bin/env python
# coding: utf-8

#  # Import library

# In[ ]:


import timm
import numpy as np
import matplotlib.pyplot as plt                        
import torch
import torchvision.models as models
import torch.nn as nn
import os,sys
import scipy.io as sio
import pdb
from time import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from scipy import spatial
from scipy.special import softmax
import torch.nn.functional as F


# # Load Data

# ### Downloading Images 
# It is recommended to download images for the desired datasets before continue running the code
# 
# Images can be downloaded via the following links:
# 
# 
# **AWA2**: https://cvml.ist.ac.at/AwA2/AwA2-data.zip
# 
# 
# **CUB**: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
# 
# 
# **SUN**: http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz
# 
# *Refer to the attached .txt file named as "Dataset_Instruction" for more information*

# ### Downloading Attributes
# For more information, refer to https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly 
# 

# 

# In[ ]:


#get_ipython().run_cell_magic('bash', '', 'if [ -d "./data" ] \nthen\n    echo "Files are already there."\nelse\n    wget -q "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"\n    unzip -q xlsa17.zip -d ./data\nfi')


# ### Choose the Dataset

# In[ ]:


DATASET = 'CUB' # ["AWA2", "CUB", "SUN"]


# Set Dataset Paths

# In[ ]:


if DATASET == 'AWA2':
  ROOT='./data/AWA2/Animals_with_Attributes2/JPEGImages/'
elif DATASET == 'CUB':
  ROOT='./data/CUB/CUB_200_2011/CUB_200_2011/images/'
elif DATASET == 'SUN':
  ROOT='./data/SUN/images/'
else:
  print("Please specify the dataset")


# In[ ]:


DATA_DIR = f'./data/xlsa17/data/{DATASET}'
data = sio.loadmat(f'{DATA_DIR}/res101.mat') 
# data consists of files names 
attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
# attrs_mat is the attributes (class-level information)
image_files = data['image_files']

if DATASET == 'AWA2':
  image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
else:
  image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])


# labels are indexed from 1 as it was done in Matlab, so 1 subtracted for Python
labels = data['labels'].squeeze().astype(np.int64) - 1
train_idx = attrs_mat['train_loc'].squeeze() - 1
val_idx = attrs_mat['val_loc'].squeeze() - 1
trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

# consider the train_labels and val_labels
train_labels = labels[train_idx]
val_labels = labels[val_idx]

# split train_idx to train_idx (used for training) and val_seen_idx
train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
# split val_idx to val_idx (not useful) and val_unseen_idx
val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]
# attribute matrix
attrs_mat = attrs_mat["att"].astype(np.float32).T

### used for validation
# train files and labels
train_files = image_files[train_idx]
train_labels = labels[train_idx]
uniq_train_labels, train_labels_based0, counts_train_labels = np.unique(train_labels, return_inverse=True, return_counts=True)
# val seen files and labels
val_seen_files = image_files[val_seen_idx]
val_seen_labels = labels[val_seen_idx]
uniq_val_seen_labels = np.unique(val_seen_labels)
# val unseen files and labels
val_unseen_files = image_files[val_unseen_idx]
val_unseen_labels = labels[val_unseen_idx]
uniq_val_unseen_labels = np.unique(val_unseen_labels)

### used for testing
# trainval files and labels
trainval_files = image_files[trainval_idx]
trainval_labels = labels[trainval_idx]
uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels, return_inverse=True, return_counts=True)
# test seen files and labels
test_seen_files = image_files[test_seen_idx]
test_seen_labels = labels[test_seen_idx]
uniq_test_seen_labels = np.unique(test_seen_labels)
# test unseen files and labels
test_unseen_files = image_files[test_unseen_idx]
test_unseen_labels = labels[test_unseen_idx]
uniq_test_unseen_labels = np.unique(test_unseen_labels)


# # Data Generator

# In[ ]:


class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root  = root
        self.image_files = image_files
        self.labels = labels 
        self.transform = transform

    def __getitem__(self, idx):
        # read the iterable image
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        # label
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)


# # Transformations

# In[ ]:


# Training Transformations
trainTransform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
# Testing Transformations
testTransform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])


# ### Average meter

# In[ ]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# # Train Function

# In[ ]:


def train(model, data_loader, train_attrbs, optimizer, use_cuda, lamb_1=1.0):
    """returns trained model"""    
    # initialize variables to monitor training and validation loss
    loss_meter = AverageMeter()
    """ train the model  """
    model.train()
    tk = tqdm(data_loader, total=int(len(data_loader)))
    for batch_idx, (data, label) in enumerate(tk):
        # move to GPU
        if use_cuda:
            data,  label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        
        x_g = model.vit(data)[0]#x_g.shape torch.size([32,1024])
        # global feature
        feat_g = model.mlp_g(x_g)
        logit_g = feat_g @ train_attrbs.T
        loss = lamb_1 * F.cross_entropy(logit_g, label)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg})
        
    # print training/validation statistics 
    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))
    

def get_reprs(model, data_loader, use_cuda):
    model.eval()
    reprs = []
    for _, (data, _) in enumerate(data_loader):
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            # only take the global feature
            feat = model.vit(data)[0]

            feat = model.mlp_g(feat)
        reprs.append(feat.cpu().data.numpy())
    reprs = np.concatenate(reprs, 0)
    return reprs

def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)

def validation(model, seen_loader, seen_labels, unseen_loader, unseen_labels, attrs_mat, use_cuda, gamma=None):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, seen_loader, use_cuda)
        unseen_reprs = get_reprs(model, unseen_loader, use_cuda)

    # Labels
    uniq_labels = np.unique(np.concatenate([seen_labels, unseen_labels]))
    updated_seen_labels = np.searchsorted(uniq_labels, seen_labels)
    uniq_updated_seen_labels = np.unique(updated_seen_labels)
    updated_unseen_labels = np.searchsorted(uniq_labels, unseen_labels)
    uniq_updated_unseen_labels = np.unique(updated_unseen_labels)
    uniq_updated_labels = np.unique(np.concatenate([updated_seen_labels, updated_unseen_labels]))

    # truncate the attribute matrix
    trunc_attrs_mat = attrs_mat[uniq_labels]
  
    #### ZSL ####
    zsl_unseen_sim = unseen_reprs @ trunc_attrs_mat[uniq_updated_unseen_labels].T
    pred_labels = np.argmax(zsl_unseen_sim, axis=1)
    zsl_unseen_predict_labels = uniq_updated_unseen_labels[pred_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, updated_unseen_labels, uniq_updated_unseen_labels)
    
    #### GZSL ####
    # seen classes
    gzsl_seen_sim = softmax(seen_reprs @ trunc_attrs_mat.T, axis=1)
    # unseen classes
    gzsl_unseen_sim = softmax(unseen_reprs @ trunc_attrs_mat.T, axis=1)

    gammas = np.arange(0.0, 1.1, 0.1)
    gamma_opt = 0
    H_max = 0
    gzsl_seen_acc_max = 0
    gzsl_unseen_acc_max = 0
    # Calibrated stacking
    for igamma in range(gammas.shape[0]):
        # Calibrated stacking
        gamma = gammas[igamma]
        gamma_mat = np.zeros(trunc_attrs_mat.shape[0])
        gamma_mat[uniq_updated_seen_labels] = gamma

        gzsl_seen_pred_labels = np.argmax(gzsl_seen_sim - gamma_mat, axis=1)
        # gzsl_seen_predict_labels = uniq_updated_labels[pred_seen_labels]
        gzsl_seen_acc = compute_accuracy(gzsl_seen_pred_labels, updated_seen_labels, uniq_updated_seen_labels)

        gzsl_unseen_pred_labels = np.argmax(gzsl_unseen_sim - gamma_mat, axis=1)
        # gzsl_unseen_predict_labels = uniq_updated_labels[pred_unseen_labels]
        gzsl_unseen_acc = compute_accuracy(gzsl_unseen_pred_labels, updated_unseen_labels, uniq_updated_unseen_labels)

        H = 2 * gzsl_seen_acc * gzsl_unseen_acc / (gzsl_seen_acc + gzsl_unseen_acc)

        if H > H_max:
            gzsl_seen_acc_max = gzsl_seen_acc
            gzsl_unseen_acc_max = gzsl_unseen_acc
            H_max = H
            gamma_opt = gamma

    print('ZSL: averaged per-class accuracy: {0:.2f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc_max * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc_max * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H_max * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma_opt))

    return gamma_opt


def test(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, use_cuda, gamma):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, test_seen_loader, use_cuda)
        unseen_reprs = get_reprs(model, test_unseen_loader, use_cuda)
    # Labels
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # ZSL
    zsl_unseen_sim = unseen_reprs @ attrs_mat[uniq_test_unseen_labels].T
    predict_labels = np.argmax(zsl_unseen_sim, axis=1)
    zsl_unseen_predict_labels = uniq_test_unseen_labels[predict_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

    # GZSL
    # seen classes
    gzsl_seen_sim = softmax(seen_reprs @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels = np.argmax(gzsl_seen_sim, axis=1)
    gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)
    
    # unseen classes
    gzsl_unseen_sim = softmax(unseen_reprs @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels = np.argmax(gzsl_unseen_sim, axis=1)
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)

    print('ZSL: averaged per-class accuracy: {0:.2f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma))


# # Data Loaders

# In[ ]:


num_workers = 0#4
### used in validation
# train data loader
train_data = DataLoader(ROOT, train_files, train_labels_based0, transform=trainTransform)
weights_ = 1. / counts_train_labels
weights = weights_[train_labels_based0]
train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=train_labels_based0.shape[0], replacement=True)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, sampler=train_sampler, num_workers=num_workers)
# seen val data loader
val_seen_data = DataLoader(ROOT, val_seen_files, val_seen_labels, transform=testTransform)
val_seen_data_loader = torch.utils.data.DataLoader(val_seen_data, batch_size=256, shuffle=False, num_workers=num_workers)
# unseen val data loader
val_unseen_data = DataLoader(ROOT, val_unseen_files, val_unseen_labels, transform=testTransform)
val_unseen_data_loader = torch.utils.data.DataLoader(val_unseen_data, batch_size=256, shuffle=False, num_workers=num_workers)

### used in testing
# trainval data loader
trainval_data = DataLoader(ROOT, trainval_files, trainval_labels_based0, transform=trainTransform)
weights_ = 1. / counts_trainval_labels
weights = weights_[trainval_labels_based0]
trainval_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=trainval_labels_based0.shape[0], replacement=True)
trainval_data_loader = torch.utils.data.DataLoader(trainval_data, batch_size=32, sampler=trainval_sampler, num_workers=num_workers)
# seen test data loader
test_seen_data = DataLoader(ROOT, test_seen_files, test_seen_labels, transform=testTransform)
test_seen_data_loader = torch.utils.data.DataLoader(test_seen_data, batch_size=256, shuffle=False, num_workers=num_workers)
# unseen test data loader
test_unseen_data = DataLoader(ROOT, test_unseen_files, test_unseen_labels, transform=testTransform)
test_unseen_data_loader = torch.utils.data.DataLoader(test_unseen_data, batch_size=256, shuffle=False, num_workers=num_workers)


# # Baseline Model (ViT)

# In[ ]:


class ViT(nn.Module):
    def __init__(self, model_name="vit_large_patch16_224_in21k", pretrained=True):
        super(ViT, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        # Others variants of ViT can be used as well
        '''
        1 --- 'vit_small_patch16_224'
        2 --- 'vit_base_patch16_224'
        3 --- 'vit_large_patch16_224',
        4 --- 'vit_large_patch32_224'
        5 --- 'vit_deit_base_patch16_224'
        6 --- 'deit_base_distilled_patch16_224',
        '''

        # Change the head depending of the dataset used 
        self.vit.head = nn.Identity()


    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  
        if self.vit.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        
        return x[:, 0], x[:, 1:]


# # Model and Optimizer Initialization

# In[ ]:


import collections
from torch import optim
use_cuda = torch.cuda.is_available()

if DATASET == 'AWA2':
  attr_length = 85
elif DATASET == 'CUB':
  attr_length = 312
elif DATASET == 'SUN':
  attr_length = 102
else:
  print("Please specify the dataset, and set {attr_length} equal to the attribute length")

vit = ViT("vit_large_patch16_224_in21k")
mlp_g = nn.Linear(1024, attr_length, bias=False)

model = nn.ModuleDict({
    "vit": vit,
    "mlp_g": mlp_g})

# finetune all the parameters
for param in model.parameters():
    param.requires_grad = True
    
# move model to GPU if CUDA is available
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam([{"params": model.vit.parameters(), "lr": 0.00001, "weight_decay": 0.0001},
                              {"params": model.mlp_g.parameters(), "lr": 0.001, "weight_decay": 0.00001}])
                              
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
#lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)


# train attributes
train_attrbs = attrs_mat[uniq_train_labels]
train_attrbs_tensor = torch.from_numpy(train_attrbs)
# trainval attributes
trainval_attrbs = attrs_mat[uniq_trainval_labels]
trainval_attrbs_tensor = torch.from_numpy(trainval_attrbs)
if use_cuda:
    train_attrbs_tensor = train_attrbs_tensor.cuda()
    trainval_attrbs_tensor = trainval_attrbs_tensor.cuda()


# In[ ]:


#model


# # Training and Testing the model

# ### Setting the calibration factor

# In[ ]:


""" Only Run this cell if you are to tune the calibration factor (gamma)
    It is data-dependent, and decided based on the validation set """
gammas = []
for i in range(20):
    train(model, train_data_loader, train_attrbs_tensor, optimizer, use_cuda, lamb_1=1.0)
    lr_scheduler.step()
    gamma = validation(model, val_seen_data_loader, val_seen_labels, val_unseen_data_loader, val_unseen_labels, attrs_mat, use_cuda)
    gammas.append(gamma)
gamma = np.mean(gammas)
print(gamma)


# ### Calibration factor is Set
# It is 0.9 for AWA2 and CUB
# 0.4 for SUN

# In[ ]:


if DATASET == 'AWA2':
  gamma = 0.9
elif DATASET == 'CUB':
  gamma = 0.9
elif DATASET == 'SUN':
  gamma = 0.4
else:
  print("Please specify the dataset, and set {attr_length} equal to the attribute length")
print('Dataset:', DATASET, '\nGamma:',gamma)


# In[ ]:


for i in range(80):
    train(model, trainval_data_loader, trainval_attrbs_tensor, optimizer, use_cuda, lamb_1=1.0)
    print(' .... Saving model ...')
    print('Epoch: ', i)
    save_path= str(DATASET) + '__ViT-ZSL__' +'Epoch_' + str(i) + '.pt'
    ckpt_path = './checkpoint/' + str(DATASET)
    path = os.path.join(ckpt_path, save_path)
    torch.save(model.state_dict(), path)

    lr_scheduler.step()
    test(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels, attrs_mat, use_cuda, gamma)


# In[ ]:




