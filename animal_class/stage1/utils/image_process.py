import torch
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import os
import numpy as np

class Mydataset(Dataset):
    def __init__(self,root_dir,csv_file,transform = None):
        self.transform = transform
        self.data_info = pd.read_csv(os.path.join(root_dir,csv_file))
        self.image = self.data_info['image'].values
        self.label = self.data_info['label'].values
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, index):
        image = cv2.imread(self.image[index])
        print(self.image[index])
        label = int(self.label[index])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample
class imagaug(object):
    def __call__(self, image):
        if np.random.uniform(0,1)>0.5:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=(0,0.2*255)),
                    iaa.Sharpen(alpha=(0.0,0.3),lightness=(0.7,1.3)),
                    iaa.Cutout(),
                    iaa.GaussianBlur(sigma=(0,1.0)),]
            )
            ])
            image = seq.augment_image(image)
        return image