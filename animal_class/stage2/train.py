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

def train(net,epoch,train_loader,optimizer,device,criterion,train_dataset):
    net.train()
    total_loss = 0.0
    iterm = 0
    dataprocess = tqdm(train_loader)
    for batch_item in dataprocess:
        image,label = batch_item['image'].to(device,torch.float32),batch_item['label'].to(device,torch.long)
        optimizer.zero_grad()
        output = net(image)
        output = output.view(-1,3)
        loss = criterion(output,label)
        total_loss += loss
        iterm+=1
        loss.backward()
        optimizer.step()
        dataprocess.set_description_str("train_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("train_loss:{:.4f}".format(loss.item()))
    avg_loss =total_loss.item() / iterm
    print("train_loss:{:.4f}".format(avg_loss))
    return avg_loss
def val(net,epoch,val_loader,device,val_dataset,criterion):
    net.eval()
    total_loss = 0.0
    dataprocess = tqdm(val_loader)
    iterm = 0
    for batch_item in dataprocess:
        image,label = batch_item['image'].to(device,torch.float32),batch_item['label'].to(device)
        output = net(image)
        output = output.view(-1, 3)
        loss = criterion(output,label)
        total_loss +=loss
        iterm+=1
        dataprocess.set_description_str("val_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("val_loss:{:.4f}".format(loss.item()))
    avg_loss= total_loss.item() / iterm
    print("val_loss:{:.4f}".format(avg_loss))
    return avg_loss
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
    for epoch in range(config.max_iter):
        trian_loss = train(net,epoch,train_loader,optimizer,device,criterion,train_dataset)
        val_loss = val(net,epoch,val_loader,device,val_dataset,criterion)
        train_epoch_loss.append(trian_loss)
        val_epoch_loss.append(val_loss)
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
if __name__ == '__main__':
    main()