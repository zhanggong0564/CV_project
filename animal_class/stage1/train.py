import torch
from stage1.utils.image_process import Mydataset,imagaug
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from stage1 import config
import os
from stage1.model.Alexnet import AlexNet
import torch.nn as nn
import numpy as np

def train(net,epoch,train_loader,optimizer,device,criterion):
    net.train()
    total_loss = 0.0
    dataprocess = tqdm(train_loader)
    for batch_item in train_loader:
        image,label = batch_item['image'].to(device,torch.float32),batch_item['label'].to(device,torch.long)
        optimizer.zero_grad()
        output = net(image)
        output = output.view(-1,2)
        loss = criterion(output,label)
        total_loss += loss
        loss.backward()
        optimizer.step()
        dataprocess.set_description_str("train_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("train_loss:{:.4f}".format(loss.item()))
    dataprocess.set_postfix_str("train_loss:{:.4f}".format(total_loss.item() / len(train_loader)))
def val(net,epoch,val_loader,device):
    net.eval()
    total_loss = 0.0
    dataprocess = tqdm(val_loader)
    for batch_item in dataprocess:
        image,label = batch_item['image'].to(device,torch.float32),batch_item['label'].to(device)
        output = net(image)
        output = output.view(-1, 2)
        loss = nn.CrossEntropyLoss(output,label)
        total_loss +=loss
        dataprocess.set_description("val_epoch:{}".format(epoch))
        dataprocess.set_postfix_str("val_loss:{:.4f}".format(loss.item()))
    dataprocess.set_postfix_str("val_loss:{:.4f}".format(total_loss.item()/len(val_loader)))
def main():
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    # device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
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
    net= AlexNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.max_iter):
        train(net,epoch,train_loader,optimizer,device,criterion)
        val(net,epoch,val_loader,device)
        if epoch%10==0:
            torch.save(net, os.path.join(os.getcwd(), config.SAVE_PATH, 'model_{}.pth'.format(epoch)))
    torch.save(net, os.path.join(os.getcwd(), config.SAVE_PATH, 'Fine_model_fine.pth'))
if __name__ == '__main__':
    main()