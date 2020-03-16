import sys
sys.path.append('/root/animal_class')
sys.path.append('/root/animal_class')
import torch
from stage3.utils.image_process import Mydataset,imagaug
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from stage3 import config
import os
from stage3.model.Alexnet import AlexNet
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def train(net,epoch,train_loader,optimizer,device,criterion1,criterion2,train_dataset):
    net.train()
    total_loss = 0.0
    iterm = 0
    total_acc1 = 0.0
    total_acc2 = 0.0
    dataprocess = tqdm(train_loader)
    for batch_item in dataprocess:
        image,label1,label2 = batch_item['image'].to(device,torch.float32),batch_item['label1'].to(device,torch.long),batch_item['label2'].to(device,torch.long),
        optimizer.zero_grad()
        output1,output2 = net(image)
        output1 = output1.view(-1,2)
        output2 = output2.view(-1,3)
        loss1 = criterion1(output1,label1)
        loss2 = criterion2(output2,label2)
        loss = 0.3*loss1+0.7*loss2
        total_loss += loss
        _,pred_m = torch.max(output1,1)
        _, pred_cls = torch.max(output2, 1)
        acc1=torch.sum(pred_m==label1)
        acc2 = torch.sum(pred_cls == label2)
        total_acc1+=acc1
        total_acc2 += acc2
        iterm+=1
        loss.backward()
        optimizer.step()
        dataprocess.set_description_str("train_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("train_loss:{:.4f}".format(loss.item()))
    avg_loss =total_loss.item() / iterm
    epoch_acc1 = total_acc1.double() / len(train_dataset)
    epoch_acc2 = total_acc2.double() / len(train_dataset)
    print("train_loss:{:.4f}".format(avg_loss))
    print('train_labe11_acc:{:.2%}'.format(epoch_acc1))
    print('train_labe12_acc:{:.2%}'. format(epoch_acc2))
    return avg_loss,epoch_acc1,epoch_acc2
def val(net,epoch,val_loader,device,val_dataset,criterion1,criterion2):
    net.eval()
    total_loss = 0.0
    iterm = 0
    total_acc1 = 0.0
    total_acc2 = 0.0
    dataprocess = tqdm(val_loader)
    iterm = 0
    for batch_item in dataprocess:
        image, label1, label2 = batch_item['image'].to(device, torch.float32), batch_item['label1'].to(device,torch.long), \
                                batch_item['label2'].to(device, torch.long)
        output1, output2 = net(image)
        output1 = output1.view(-1,2)
        output2 = output2.view(-1,3)
        loss1 = criterion1(output1,label1)
        loss2 = criterion2(output2,label2)
        loss = 0.3*loss1+0.7*loss2
        total_loss += loss
        _,pred_m = torch.max(output1,1)
        _, pred_cls = torch.max(output2, 1)
        acc1=torch.sum(pred_m==label1)
        acc2 = torch.sum(pred_cls == label2)
        total_acc1+=acc1
        total_acc2 += acc2
        iterm+=1
        dataprocess.set_description_str("val_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("val_loss:{:.4f}".format(loss.item()))
    avg_loss= total_loss.item() / iterm
    epoch_acc1 = total_acc1.double() / len(val_dataset)
    epoch_acc2 = total_acc2.double() / len(val_dataset)
    print("val_loss:{:.4f}".format(avg_loss))
    print('val_labe11_acc:{:.2%}'.format(epoch_acc1))
    print('val_labe12_acc:{:.2%}'.format(epoch_acc2))
    return avg_loss,epoch_acc1,epoch_acc2
def main():
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    print('runing in the {}'.format(device))
    transform = transforms.Compose([
        imagaug(),
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.ToTensor()
    ])
    train_dataset = Mydataset(config.root_dir, config.train_csv_file, transform=transform)
    val_dataset = Mydataset(config.root_dir, config.val_csv_file, transform=transforms.Compose([transforms.ToPILImage(),
                                                                                                transforms.Resize((200,200)),
                                                                                                transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,**kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,**kwargs)
    net= AlexNet(2,3).to(device)
    net = nn.DataParallel(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr,weight_decay=0.02)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()
    train_epoch_loss= []
    val_epoch_loss = []
    train_acc_list1 = []
    train_acc_list2 = []
    val_acc_list1 = []
    val_acc_list2 = []
    for epoch in range(config.max_iter):
        trian_loss ,train_acc1,train_acc2= train(net,epoch,train_loader,optimizer,device,criterion1,criterion2,train_dataset)
        val_loss ,val_acc1,val_acc2 = val(net,epoch,val_loader,device,val_dataset,criterion1,criterion2)
        train_epoch_loss.append(trian_loss)
        val_epoch_loss.append(val_loss)
        train_acc_list1.append(train_acc1)
        train_acc_list2.append(train_acc2)
        val_acc_list1.append(val_acc1)
        val_acc_list2.append(val_acc2)
        if epoch%10==0:
            print('save model')
            torch.save(net, os.path.join(os.getcwd(), config.SAVE_PATH, 'model_{}.pth'.format(epoch)))
    torch.save(net, os.path.join(os.getcwd(), config.SAVE_PATH, 'Fine_model_fine.pth'))
    x = range(0,config.max_iter)
    y1 = train_epoch_loss
    y2 = val_epoch_loss
    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
    plt.legend()
    plt.title('train and val loss vs epoches')
    plt.ylabel('loss')
    plt.savefig("train and val loss vs epoches.jpg")
    plt.close('all')  # 关闭图 0
    y5 = train_acc_list1
    y6= val_acc_list1
    plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
    plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
    plt.legend()
    plt.title('train and val Classes_acc vs. epoches')
    plt.ylabel('Classes_accuracy')
    plt.savefig("train and val Classes_acc vs epoches.jpg")
    y8 = train_acc_list2
    y9= val_acc_list2
    plt.plot(x, y8, color="r", linestyle="-", marker=".", linewidth=1, label="train")
    plt.plot(x, y9, color="b", linestyle="-", marker=".", linewidth=1, label="val")
    plt.legend()
    plt.title('train and val Classes_acc vs. epoches')
    plt.ylabel('Classes_accuracy')
    plt.savefig("train and val Classes_acc vs epoches.jpg")
    plt.close('all')
if __name__ == '__main__':
    main()