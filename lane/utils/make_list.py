import os
import pandas as pd
from sklearn.utils import shuffle

image_list= []
label_list=[]

image_dir= r'D:\kkb\lane\data\ColorImage'
label_dir = r'D:\kkb\lane\data\Label'
data_list = r'D:\kkb\lane\data_list'

for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir,s1)
    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1,s2)
        for s3 in os.listdir(image_sub_dir2):
            image_sub_dir3 = os.path.join(image_sub_dir2,s3)
            for s4 in os.listdir(image_sub_dir3):
                image_sub_dir4 = os.path.join(image_sub_dir3,s4)
                if not os.path.exists(image_sub_dir4):
                    print(image_sub_dir4)
                    continue
                image_list.append(image_sub_dir4)
for s1 in os.listdir(label_dir):
    label_sub_dir1 = os.path.join(label_dir,s1)
    for s2 in os.listdir(label_sub_dir1):
        label_sub_dir2 = os.path.join(label_sub_dir1,s2)
        for s3 in os.listdir(label_sub_dir2):
            label_sub_dir3 = os.path.join(label_sub_dir2,s3)
            for s4 in os.listdir(label_sub_dir3):
                label_sub_dir4 = os.path.join(label_sub_dir3,s4)
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)
                    continue
                label_list.append(label_sub_dir4)
assert len(image_list)==len(label_list)
print('The length of image dataset is {},The length of label dataset is {}'.format(len(image_list),len(label_list)))
#划分训练集和验证集、测试集
total = len(image_list)
six_total = int(total*0.6)
eight_total = int(total*0.8)
all = pd.DataFrame({'image':image_list,'label':label_list})
all_shuffle = shuffle(all)
train_dataset = all_shuffle[:six_total]
val_dataset = all_shuffle[six_total:eight_total]
test_dataset = all_shuffle[eight_total:]

if not os.path.exists(data_list):
    os.makedirs(data_list)
train_info_path = os.path.join(data_list,'train.csv')
val_info_path = os.path.join(data_list,'val.csv')
test_info_path = os.path.join(data_list,'test.csv')

train_dataset.to_csv(train_info_path,index=False)
val_dataset.to_csv(val_info_path,index=False)
test_dataset.to_csv(test_info_path,index=False)

print('finish')