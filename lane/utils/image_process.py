import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from utils.process_label import encode_labels,decode_labels
import numpy as np
from imgaug import augmenters as iaa
import random

def crop_resize_data(image,label=None,image_size=(1024,384),offset=690):
    roi_image = image[offset:,:]
    if label is not None:
        roi_label = label[offset:,:]
        train_image = cv2.resize(roi_image,image_size,interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label,image_size,interpolation=cv2.INTER_NEAREST)
        return train_image,train_label
    else:
        train_image= cv2.resize(roi_image,image_size,interpolation=cv2.INTER_LINEAR)
        return train_image
class LaneDateset(Dataset):
    def __init__(self,root_dir,csv_file,transform = None):
        self.data = pd.read_csv(os.path.join(root_dir,csv_file))
        self.image = self.data['image'].values
        self.labels = self.data['label'].values
        self.transform= transform
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, index):
        ori_image =cv2.imread(self.image[index])
        ori_mask = cv2.imread(self.labels[index],cv2.IMREAD_GRAYSCALE)
        train_image,train_mask = crop_resize_data(ori_image,ori_mask)
        train_mask = encode_labels(train_mask)
        sample ={'image':train_image.copy(),'label':train_mask.copy()}
        if self.transform:
            sample = self.transform(sample)
        return sample
class ImageAug(object):
    def __call__(self, sample):
        image, mask = sample['image'],sample['label']
        if np.random.uniform(0,1) >0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0,0.2*255)),
                iaa.Sharpen(alpha=(0.0,0.3),lightness=(0.7,1.3)),
                iaa.Cutout(),
                iaa.GaussianBlur(sigma=(0,1.0))
            ])])
            image= seq.augment_image(image)
        sample = {'image':image,'label':mask}
        return sample
class DeformAug(object):
    def __call__(self, sample):
        image, mask = sample['image'],sample['label']
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
        seg_to = seq.to_deterministic()
        image = seg_to.augment_image(image)
        mask = seg_to.augment_image(mask)
        sample = {'image':image,'label':mask}
        return sample
class ScaleAug(object):
    def __call__(self, sample):
        image, mask = sample['image'],sample['label']
        scale = random.uniform(0.7, 1.5)
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()
        aug_image = cv2.resize(aug_image, (int (scale * w), int (scale * h)))
        aug_mask = cv2.resize(aug_mask, (int (scale * w), int (scale * h)))
        if (scale < 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            aug_image = np.pad(aug_image, pad_list, mode="constant")
            aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant")
        if (scale > 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_crop = int ((new_h - h) / 2)
            pre_w_crop = int ((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            aug_image = aug_image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            sample = {'image':aug_image,'label':aug_mask}
        return sample
class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample['image'],sample['label']
        image = np.transpose(image,(2,0,1))
        image = image.astype(np.int32)
        mask = mask.astype(np.uint8)
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy())}

