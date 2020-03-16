from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDateset,ToTensor,ImageAug,DeformAug,ScaleAug
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import config
import cv2
from model.deeplapv3plus import DeeplabV3Plus
import os
import torch.nn.functional as F
from utils.metric import compute_iou
from utils.loss import MySoftmaxCrossEntropyLoss



def train(net,epoch,dataloader,optimizer,trainF,device):
    net.train()
    total_loss= 0.0
    dataprocess = tqdm(dataloader)
    for batch_item in dataprocess:
        image,mask = batch_item['image'].to(device,torch.float32),batch_item['mask'].to(device,torch.long)
        optimizer.zero_grad()
        output = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(output,mask)

        total_loss+=mask_loss
        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    trainF.write("Epoch:{},mask loss is {:.4f}".format(epoch,total_loss/len(dataloader)))
    trainF.flush()
def test(net,epoch,dataloader,testF,device):
    net.eval()
    total_loss = 0.0
    dataprocess = tqdm(dataloader)
    result = {'TP':{i:0 for i in range(8)},'TA':{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'].to(device,torch.float32), batch_item['mask'].to(device,torch.long)
        output = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=config.NUM_CLASSES)(output,mask)
        total_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(output,dim=1),dim=1)
        result = compute_iou(pred,mask,result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str('mask_loss:{:.4f}'.format(mask_loss))
    testF.write('Epoch:{}'.format(epoch))
    for i in range(8):
        result_string = "{}: {:.4f} \n".format(i, result["TP"][i]/result["TA"][i])
        print(result_string)
        testF.write(result_string)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_loss / len(dataloader)))
    testF.flush()
def main():
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    trainF = open(os.path.join(config.SAVE_PATH,'train.csv'),'w')
    testF = open(os.path.join(config.SAVE_PATH, "test.csv"), 'w')
    kwargs= {'num_workers':4,'pin_memory':True} if torch.cuda.is_available() else {}
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('runing in the {}'.format(device))

    transform = transforms.Compose([ImageAug(),DeformAug(),ScaleAug(),ToTensor()])
    train_dataset =LaneDateset(config.Root_dir,config.train_csv_dir,transform= transform)
    val_dataset = LaneDateset(config.Root_dir,config.val_csv_dir,transform = transforms.Compose([ToTensor()]))
    train_loader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,drop_last=True,**kwargs)
    val_loader = DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False,drop_last=False,**kwargs)
    net = DeeplabV3Plus(config).to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=config.BASE_LR,weight_decay=config.WEIGHT_DECAY)
    for epoch in range(config.EPOCHS):
        train(net,epoch,train_loader,optimizer,trainF,device)
        test(net,epoch,val_loader,testF,device)
        if epoch%10==0:
            torch.save(net,os.path.join(os.getcwd(),config.SAVE_PATH,'lane_model_{}.pth'.format(epoch)))
    trainF.close()
    testF.close()
    torch.save(net,os.path.join(os.getcwd(),config.SAVE_PATH,'lane_model_fine.pth'))

if __name__ == '__main__':
    main()
    # root_dir= r'D:\kkb\lane\data_list'
    # train_csv_dir = 'train.csv'
    # kwargs= {'num_worker':4,'pin_memory':True} if torch.cuda.is_available() else {}
    # device= torch.device('cpu')
    # print('runing in the cpu')
    # transform = transforms.Compose([ImageAug(),DeformAug(),ScaleAug(),ToTensor()])
    # train_dataset =LaneDateset(root_dir,train_csv_dir,transform= transform)
    # train_data_batch = DataLoader(train_dataset,batch_size=1,shuffle=True,drop_last=True,pin_memory=True,num_workers=4)
    # dataprocess = tqdm(train_data_batch)
    # # dataprocess = train_data_batch
    # # optimizer= torch.optim.Adam()
    # for batch_item in dataprocess:
    #     image,mask = batch_item['image'],batch_item['mask']
    #     # image,mask = image.to(device),mask.to(device)
    #     print(image.size(),mask.size())
    #     image = image.numpy()
    #     print(type(image))
    #     plt.imshow(np.transpose(image[0],(1,2,0)).astype('uint8'))
    #     plt.show()
    # train()
