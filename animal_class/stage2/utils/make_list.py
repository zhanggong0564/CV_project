import os
import pandas as pd
import torch.nn as nn
train_image_list = []
train_label_list = []
val_image_list=[]
val_label_list = []

train_image_dir = '/root/animal_class/data/train'
data_list_dir = '/root/animal_class/stage2/data_list'
val_image_dir = '/root/animal_class/data/val'
for s1 in os.listdir(train_image_dir):
    if s1 == 'chickens':
        image_sub_dir = os.path.join(train_image_dir,s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir,s2)
            train_image_list.append(image_sub_dir1)
            train_label_list.append(0)
    elif s1=='rabbits':
        image_sub_dir = os.path.join(train_image_dir, s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir, s2)
            train_image_list.append(image_sub_dir1)
            train_label_list.append(1)
    else:
        image_sub_dir = os.path.join(train_image_dir, s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir, s2)
            train_image_list.append(image_sub_dir1)
            train_label_list.append(2)
for s1 in os.listdir(val_image_dir):
    if s1 == 'chickens':
        image_sub_dir = os.path.join(val_image_dir,s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir,s2)
            val_image_list.append(image_sub_dir1)
            val_label_list.append(0)
    elif s1=='rabbits':
        image_sub_dir = os.path.join(val_image_dir, s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir, s2)
            val_image_list.append(image_sub_dir1)
            val_label_list.append(1)
    else:
        image_sub_dir = os.path.join(val_image_dir, s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir, s2)
            val_image_list.append(image_sub_dir1)
            val_label_list.append(2)
train = pd.DataFrame({'image':train_image_list,'label':train_label_list})
val = pd.DataFrame({'image':val_image_list,'label':val_label_list})
if not os.path.exists(data_list_dir):
    os.makedirs(data_list_dir)
train_info_dir = os.path.join(data_list_dir,'train.csv')
val_info_dir = os.path.join(data_list_dir,'val.csv')
train.to_csv(train_info_dir,index=False)
val.to_csv(val_info_dir,index=False)
print('Finish')
