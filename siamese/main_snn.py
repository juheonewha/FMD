import torch
import numpy as np
import torchvision.transforms as transforms
from siamese import *
from torch.utils.data import Dataset, DataLoader
from snn import *
from sklearn import metrics
import argparse

def train_snn(net,training_loader, criterion,
              optimizer, epoch, n_epochs):
    net.train()
   
    running_loss = 0.0

    print('Training----->')
    for i, batch in enumerate(training_loader):
        inputs1, inputs2, labels = batch
        inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

        optimizer.zero_grad()

     
        outputs = net(inputs1, inputs2)
        outputs = outputs.squeeze()
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' % (epoch + 1, n_epochs, i + 1, len(training_loader),
                                                                loss.item()))


    training_loss = running_loss / len(training_loader)
    return training_loss


def test_snn(net,test_loader):

    net.eval()
    
    y_true = None
    y_pred = None

    print('Testing----->')
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs1, inputs2, labels = batch

            if y_true is None:
                y_true = labels
            else:
                y_true = torch.cat((y_true, labels), 0)

            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()
            # Forward
            outputs = net(inputs1, inputs2)
            outputs = torch.sigmoid(outputs)

            if y_pred is None:
                y_pred = outputs.cpu()
            else:
                y_pred = torch.cat((y_pred, outputs.cpu()), 0)

    y_pred = y_pred.squeeze()
    return y_true, y_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SNN Training')
    parser.add_argument('--input1', type=str, required=True, help='Path to input1 dataset')
    parser.add_argument('--input2', type=str, required=True, help='Path to input2 dataset')
    parser.add_argument('--ipc', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_class', type=int, default=4, help='number of class')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for training SNN')
    parser.add_argument('--save_dir', type=str, default='./result', help='Path to save net parameters')
    args = parser.parse_args()

    train_dataset=SiameseDataset(input1=args.input1,ipc=args.ipc,input2=args.input2,n_class=args.num_class,state='train')
    
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    net=SiameseNetwork().cuda()
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001) 
    criterion=nn.BCEWithLogitsLoss().cuda()
    n_epochs=args.epochs
    for epoch in range(n_epochs):
        train_loss=train_snn(net,train_loader,criterion,optimizer,epoch,n_epochs)
        torch.save(net.state_dict(),args.save_dir+'/{}.pth'.format(epoch))
    test_dataset=SiameseDataset(input1=args.input1,ipc=args.ipc,input2=args.input2,n_class=args.num_class,state='test')
    test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    
    test_net=SiameseNetwork().to('cuda')
    test_net.load_state_dict(torch.load(args.save_dir+'/{}.pth'.format(n_epochs-1)))
    y_true,y_pred=test_snn(test_net,test_loader)
    auc = metrics.roc_auc_score(y_true, y_pred)
    print('AUC for SNN : {}'.format(auc))