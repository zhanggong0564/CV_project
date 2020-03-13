import sys
sys.path.append('/root/animal_class')
sys.path.append('/root/animal_class')
import torch
from stage1.model.Alexnet import AlexNet
import cv2
import numpy as np
from stage1.utils.image_process import Mydataset
import stage1.config as config
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
def img_transform(img):
    img = np.transpose(img,(2,1,0))
    img = img[np.newaxis,...].astype(np.float32)
    img = torch.from_numpy(img.copy())
    return img
def val_acc(val_dataloader,device,model,val_dataset):
    model.eval()
    data_process = tqdm(val_dataloader)
    total_acc =0.0
    for item in data_process:
        image,label = item['image'].to(device,torch.float32),item['label'].to(device,torch.long)
        pre = model(image)
        pred = pre.view(-1,2)
        _,pred_cls = torch.max(pred,1)
        acc=torch.sum(pred_cls==label)
        total_acc+=acc
        data_process.set_description_str("loading image")
        data_process.set_postfix_str("acc:{:.2%}".format(acc.double()/128))
    epoch_lacc = total_acc.double()/len(val_dataset)
    print("acc:{:.2%}".format(epoch_lacc))
def main():
    val_dataset = Mydataset(config.root_dir,config.val_csv_file,transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]))
    val_dataloader = DataLoader(val_dataset,batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_path = '/root/animal_class/stage1/logs/Fine_model_fine.pth'
    model = torch.load(mode_path).to(device)
    val_acc(val_dataloader,device,model,val_dataset)
if __name__ == '__main__':
    main()