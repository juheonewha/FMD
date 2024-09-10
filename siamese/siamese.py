import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

import time
from cv2 import mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import glob
import natsort
import torch
from PIL import Image

'''for covid(4 class)'''
class SiameseDataset(Dataset):
    def __init__(self,input1,input2,ipc,n_class,state='train'):

        self.state=state
        r_t=int(ipc*0.7)

        self.input1=input1
        self.input1_list=torch.load(self.input1)
        a=self.input1_list[:ipc*1]
        b=self.input1_list[ipc*1:ipc*2]
        c=self.input1_list[ipc*2:ipc*3]
        d=self.input1_list[ipc*3:ipc*4]

       
        self.train_list=torch.cat((a[:r_t],b[:r_t],c[:r_t],d[:r_t]),dim=0)
        self.test_list=torch.cat((a[r_t:],b[r_t:],c[r_t:],d[r_t:]),dim=0)
        
        self.input2=input2
        self.input2_list=torch.load(self.input2)
        
        e = self.input2_list[:ipc*1]
        f = self.input2_list[ipc*1:ipc*2]
        g = self.input2_list[ipc*2:ipc*3]
        h = self.input2_list[ipc*3:ipc*4]

        
        self.train2_list=torch.cat((e[:r_t],g[:r_t],f[:r_t],h[:r_t]),dim=0)
       
        self.test2_list=torch.cat((e[r_t:],g[r_t:],f[r_t:],h[r_t:]),dim=0)

        k=[1 for _ in range(r_t)]
        l=[0 for _ in range(r_t)]
        m=[1 for _ in range(ipc-r_t)]
        n=[0 for _ in range(ipc-r_t)]
        if self.state=='train':
            self.label= l + k  + k + l 
        else:
            self.label= n + m + m + n

  
    
    def __getitem__(self,idx):

        if self.state=='train':
            img1=self.train_list[idx]
            img2=self.train2_list[idx]
            label=self.label[idx]
        else:
            img1=self.test_list[idx]
            img2=self.test2_list[idx]
            label=self.label[idx]
        '''for 1 channel image'''
        img1=self.input1_list[idx]
        img2=self.input2_list[idx]
        img1=img1.unsqueeze(0)
        img1=img1.repeat(1,3,1,1)
        img1=img1.squeeze(0)
        img2=img2.unsqueeze(0)
        img2=img2.repeat(1,3,1,1)
        img2=img2.squeeze(0)

        return img1,img2,label

    def __len__(self):
        return len(self.label)





