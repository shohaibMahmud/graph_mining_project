#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:47:57 2021

@author: Zongyu Li
"""
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import scipy.io
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_recall_fscore_support
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



# need to change the dataloader and the TimeseriesData_train_test to give consistent input as the video


# create dataloader 
def load_data(gesture,random_state,data_dir=None):
    '''
    load the data and remember to use the gesture=,random_state=,data_dir=

    Parameters
    ----------
    gesture : Stirng
        the gesture.
    random_state : int
        A number for splitting the data.
    data_dir : 'string'
        DESCRIPTION. The default is None.

    Returns
    -------
        (x_train, x_test, y_train,y_test).
    '''
    mat = scipy.io.loadmat(data_dir)
    cur = mat[gesture]
    init_x = cur[:,0]
    init_y = cur[:,2]
    x_train, x_test, y_train,y_test = train_test_split(init_x,init_y,test_size=0.2,random_state=random_state)
    return (x_train, x_test, y_train,y_test)




class TimeseriesData_train_test(Dataset):
    def __init__(self, gesture,x_train,y_train, win_len=1, stride=10):
        '''
        The data loader

        Parameters
        ----------
        gesture : String
            the gesture.
        x_train : object array for each entry[length,fatures]
            training kinematics.
        y_train : object array for each entry[class]
            error-1, normal -0.
        win_len : int
            The length of the sliding window, and the input to the network. The default is 1.
        stride : int
            the stride. The default is 10.

        Returns
        -------
        None.

        '''
        
        ## on win_len can put in len of different sizes based on the data distribution, stride could be based on
        ## std 

        init_x = x_train
        init_y  =y_train

        # use a sliding window to create more data per trial
        self.L=[]
        self.Lx=[]
        self.R=[]
        self.Rx=[]
        self.y=[]
        for idx,data in enumerate(init_x):
            time_len = data.shape[0]
            start = (time_len-win_len)%stride
            y_val=init_y[idx]
            mean_f = np.mean(data,0)
            std_f = np.std(data,0)
            data = (data-mean_f)/std_f
            L_data = data[:,3:13]
            Lx_data = data[:,0:3]
            R_data = data[:,16:26]
            Rx_data = data[:,13:16]
            cur_data_L=[L_data[i:i+win_len,:].T for i in \
                      np.arange(start,time_len-win_len+stride,stride) ]
            cur_data_Lx=[Lx_data[i:i+win_len,:].T for i in \
                      np.arange(start,time_len-win_len+stride,stride) ]
            cur_data_R=[R_data[i:i+win_len,:].T for i in \
            np.arange(start,time_len-win_len+stride,stride) ]
            cur_data_Rx=[Rx_data[i:i+win_len,:].T for i in \
            np.arange(start,time_len-win_len+stride,stride) ]
            cur_y=np.repeat(y_val, len(np.arange(start,time_len-win_len+stride,stride))) 
            for i,seq in enumerate(cur_data_L):
                # count_zero=sum(np.array(seq[0,:])==0)/win_len
                # if count_zero<0.3:
                self.L.append(seq)
                self.Lx.append(cur_data_Lx[i])
                self.y.append(cur_y[i])
                self.R.append(cur_data_R[i])
                self.Rx.append(cur_data_Rx[i])
        self.y = [val=='err' for val in self.y]
        self.y = np.array(self.y, dtype=np.float32)
        self.L = np.array(self.L, dtype=np.float32)
        self.R = np.array(self.R, dtype=np.float32)
        self.seq_len = len(self.y)
        self.stride = stride
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.L[index],self.Lx[index],self.R[index],self.Rx[index],self.y[index]


class TimeseriesNet_light(nn.Module):
    def __init__(self):
        super(TimeseriesNet_light,self).__init__()
        # self.festures=13
        self.seq_len =60
        # self.hidden_dim = 1024
        # self.layer_dim =1
        self.stage_1_conv_x=nn.Conv1d(3,64,kernel_size=5,stride=2)
        self.stage_1_pool_x = nn.MaxPool1d(2,2)
        self.stage_1_drop_x = nn.Dropout(p=0.2)
        self.stage_1_norm_x = nn.BatchNorm1d(64)
        self.stage_2_conv_x = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool_x = nn.MaxPool1d(2,2)
        self.stage_2_drop_x = nn.Dropout(p=0.2)
        self.stage_2_norm_x = nn.BatchNorm1d(128)
        self.linear1_x = nn.Linear(256,168)
        self.linear2_x = nn.Linear(168,60)
        self.linear3_x = nn.Linear(60,32)
        
        
        
        self.stage_1_conv_x2=nn.Conv1d(3,64,kernel_size=5,stride=2)
        self.stage_1_pool_x2 = nn.MaxPool1d(2,2)
        self.stage_1_drop_x2 = nn.Dropout(p=0.2)
        self.stage_1_norm_x2 = nn.BatchNorm1d(64)
        self.stage_2_conv_x2 = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool_x2 = nn.MaxPool1d(2,2)
        self.stage_2_drop_x2 = nn.Dropout(p=0.2)
        self.stage_2_norm_x2 = nn.BatchNorm1d(128)
        self.linear1_x2 = nn.Linear(256,168)
        self.linear2_x2 = nn.Linear(168,60)
        self.linear3_x2 = nn.Linear(60,32)
        
        
        self.stage_1_conv = nn.Conv1d(10,64,kernel_size=5,stride=2)
        self.stage_1_pool = nn.MaxPool1d(2,2)
        self.stage_1_drop = nn.Dropout(p=0.2)
        self.stage_1_norm = nn.BatchNorm1d(64)
        self.stage_2_conv = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool = nn.MaxPool1d(2,2)
        self.stage_2_drop = nn.Dropout(p=0.2)
        self.stage_2_norm = nn.BatchNorm1d(128)
        
        
        self.stage_1_conv_2 = nn.Conv1d(10,64,kernel_size=5,stride=2)
        self.stage_1_pool_2 = nn.MaxPool1d(2,2)
        self.stage_1_drop_2 = nn.Dropout(p=0.2)
        self.stage_1_norm_2 = nn.BatchNorm1d(64)
        self.stage_2_conv_2 = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.stage_2_pool_2 = nn.MaxPool1d(2,2)
        self.stage_2_drop_2 = nn.Dropout(p=0.2)
        self.stage_2_norm_2 = nn.BatchNorm1d(128)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(256,168)
        # self.linear2 = nn.Linear(336,168)
        self.linear2 = nn.Linear(168,60)
        self.linear3 = nn.Linear(60,32)
        
        self.linear1_2 = nn.Linear(256,168)
        # self.linear2 = nn.Linear(336,168)
        self.linear2_2 = nn.Linear(168,60)
        self.linear3_2 = nn.Linear(60,32)
        ## linear for prediction evaluation 
        # self.linear_p1 = nn.Linear(120,l1)
        # self.linear_p2 = nn.Linear(l1,l2)
        # self.linear_p3 = nn.Linear(l2,1)
        self.initialize_weights()
        
        #barch normalization
        #self.stage_2_conv = nn.Conv1d()
    def forward(self,l,lx,r,rx):

        
        l = F.relu(self.stage_1_conv(l))
        l = self.stage_1_pool(l)
        l = self.stage_1_drop(l)
        l = self.stage_1_norm(l)
        l = F.relu(self.stage_2_conv(l))
        l = self.stage_2_pool(l)
        l = self.stage_2_drop(l)
        l = self.stage_2_norm(l)
        l = self.flat(l)
        l = F.relu(self.linear1(l))
        l = F.relu(self.linear2(l))
        l = F.relu(self.linear3(l))
        
        lx = F.relu(self.stage_1_conv_x(lx))
        lx = self.stage_1_pool_x(lx)
        lx = self.stage_1_drop_x(lx)
        lx = self.stage_1_norm_x(lx)
        lx = F.relu(self.stage_2_conv_x(lx))
        lx = self.stage_2_pool_x(lx)
        lx = self.stage_2_drop_x(lx)
        lx = self.stage_2_norm_x(lx)
        lx = self.flat(lx)
        lx = F.relu(self.linear1_x(lx))
        lx = F.relu(self.linear2_x(lx))
        lx = F.relu(self.linear3_x(lx))
  
          
        r = F.relu(self.stage_1_conv_2(r))
        r = self.stage_1_pool_2(r)
        r = self.stage_1_drop_2(r)
        r = self.stage_1_norm_2(r)
        r = F.relu(self.stage_2_conv_2(r))
        r = self.stage_2_pool_2(r)
        r = self.stage_2_drop_2(r)
        r = self.stage_2_norm_2(r)
        r = self.flat(r)
        r = F.relu(self.linear1_2(r))
        r = F.relu(self.linear2_2(r))
        r = F.relu(self.linear3_2(r))
        
        rx = F.relu(self.stage_1_conv_x2(rx))
        rx = self.stage_1_pool_x2(rx)
        rx = self.stage_1_drop_x2(rx)
        rx = self.stage_1_norm_x2(rx)
        rx = F.relu(self.stage_2_conv_x2(rx))
        rx = self.stage_2_pool_x2(rx)
        rx = self.stage_2_drop_x2(rx)
        rx = self.stage_2_norm_x2(rx)
        rx = self.flat(rx)
        rx = F.relu(self.linear1_x2(rx))
        rx = F.relu(self.linear2_x2(rx))
        rx = F.relu(self.linear3_x2(rx))

        
        # comb = torch.cat((l,r),dim=1)    
        # val = F.relu(self.linear_p1(comb))
        # val = F.relu(self.linear_p2(val))
        # val = self.linear_p3(val)
        
        return (l,lx,r,rx)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)
