import sys
sys.path.append('/root/animal_class')
sys.path.append('/root/animal_class')
import torch
from stage2.utils.image_process import Mydataset,imagaug
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from stage2 import config
import os
from stage2.model.Alexnet import AlexNet
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def train(net,epoch,train_loader,optimizer,device,criterion,train_dataset):
    net.train()
    total_loss = 0.0
    iterm = 0
    total_acc = 0.0
    dataprocess = tqdm(train_loader)
    for batch_item in dataprocess:
        image,label = batch_item['image'].to(device,torch.float32),batch_item['label'].to(device,torch.long)
        optimizer.zero_grad()
        output = net(image)
        output = output.view(-1,3)
        loss = criterion(output,label)
        total_loss += loss
        _,pred_cls = torch.max(output,1)
        acc=torch.sum(pred_cls==label)
        iterm+=1
        loss.backward()
        optimizer.step()
        dataprocess.set_description_str("train_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("train_loss:{:.4f},acc{:.2%}".format(loss.item(),acc))
    avg_loss =total_loss.item() / iterm
    epoch_acc = total_acc.double() / len(train_dataset)
    print("train_loss:{:.4f}".format(avg_loss))
    return avg_loss,epoch_acc
def val(net,epoch,val_loader,device,val_dataset,criterion):
    net.eval()
    total_loss = 0.0
    total_acc = 0.0
    dataprocess = tqdm(val_loader)
    iterm = 0
    for batch_item in dataprocess:
        image,label = batch_item['image'].to(device,torch.float32),batch_item['label'].to(device)
        output = net(image)
        output = output.view(-1, 3)
        loss = criterion(output,label)
        total_loss +=loss
        _,pred_cls = torch.max(output,1)
        acc=torch.sum(pred_cls==label)
        total_acc +=acc
        iterm+=1
        dataprocess.set_description_str("val_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("val_loss:{:.4f},acc{:.2%}".format(loss.item(),acc/len(val_loader)))
    avg_loss= total_loss.item() / iterm
    print("val_loss:{:.4f}".format(avg_loss))
    epoch_acc = total_acc.double() / len(val_dataset)
    return avg_loss,epoch_acc
def main():
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    net= AlexNet(3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    criterion = nn.NLLLoss()
    train_epoch_loss= []
    val_epoch_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(config.max_iter):
        trian_loss ,train_acc = train(net,epoch,train_loader,optimizer,device,criterion,train_dataset)
        val_loss ,val_acc = val(net,epoch,val_loader,device,val_dataset,criterion)
        train_epoch_loss.append(trian_loss)
        val_epoch_loss.append(val_loss)
        train_acc.append(train_acc)
        val_acc.append(val_acc)
        if epoch%10==0:
            torch.save(net, os.path.join(os.getcwd(), config.SAVE_PATH, 'model_{}.pth'.format(epoch)))
    torch.save(net, os.path.join(os.getcwd(), config.SAVE_PATH, 'Fine_model_fine.pth'))
    x = range(0,100)
    y1 = train_epoch_loss
    y2 = val_epoch_loss
    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
    plt.legend()
    plt.title('train and val loss vs epoches')
    plt.ylabel('loss')
    plt.savefig("train and val loss vs epoches.jpg")
    plt.close('all')  # 关闭图 0
    y5 = train_acc
    y6= val_acc
    plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
    plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
    plt.legend()
    plt.title('train and val Classes_acc vs. epoches')
    plt.ylabel('Classes_accuracy')
    plt.savefig("train and val Classes_acc vs epoches.jpg")
    plt.close('all')
if __name__ == '__main__':
    main()