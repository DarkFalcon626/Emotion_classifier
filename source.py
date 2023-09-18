# -*- coding: utf-8 -*-
"""
    Emotion Classification
    -------------------------
    @author: Andrew Francey
    @date: 9/12/2023
    -------------------------
    source.py
    import source as src
"""

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import os


class Data():
    
    def __init__(self, data_path):
        
        self.emotions = os.listdir(data_path + '/test')
        
        test_data = []
        test_target = []
        
        train_data = []
        train_target = []
        
        ## Converts the images to numpy arrays and assigns the target a value
        for mode in ['test', 'train']:
            i = 0 # Counter coresponding to assigning the emotion a value
            path = data_path+'/'+mode # Data path for the folders with images
            for name in self.emotions:
                for img in os.listdir(path+'/'+name):
                    image = Image.open(path+'/'+name+'/'+img)
                    self.res = image.size[0]
                    image_array = np.asarray(image) # Convert to numpy array
                    if mode == 'test':
                        test_data.append(image_array)
                        test_target.append(i)
                    else:
                        train_data.append(image_array)
                        train_target.append(i)
                i += 1
        
        
        ## Determine the size of the test and training sets
        self.test_size = len(test_data)
        self.train_size = len(train_data)
        
        ## Convert our lists into arrays
        test_data = np.array(test_data) 
        train_data = np.array(train_data)
        test_target = np.array(test_target)
        train_target = np.array(train_target)
        
        ## add an extra dimension to our array
        test_data = test_data.reshape(self.test_size,1,self.res,self.res)
        train_data = train_data.reshape(self.train_size,1,self.res,self.res)
        
        ## Convert to tensors
        self.test_data = torch.tensor(test_data)
        self.train_data = torch.tensor(train_data)
        test_target = torch.tensor(test_target)
        train_target = torch.tensor(train_target)
        
        self.num_emotions = len(self.emotions) #Determine the number of emotions in data_set
        
        ## Convert the targets to vectors
        self.test_target = func.one_hot(test_target.to(torch.int64),
                                        num_classes = self.num_emotions)
        self.train_target = func.one_hot(train_target.to(torch.int64), 
                                         num_classes = self.num_emotions)
        
        
    def __getitem__(self,index,train=True):
        if train:
            target = self.train_target[index]
            value = self.train_data[index]
        else:
            target = self.test_target[index]
            value = self.test_data[index]
        return value, target
 
    
def conv_width(w, p, k, s):
    return 1 + (w + (2*p)-k)//s

def pool_width(w, p, k, s):
    return 1 + (w + (2*p)-(k-1))//s

class Net(nn.Module):
    
    def __init__(self, data, net_params):
        super(Net, self).__init__()
        
        self.emotion_lst = data.emotions
        
        k1 = net_params['kernal_1']
        p1 = net_params['padding_1']
        s1 = net_params['stride_1']
        
        width2 = conv_width(data.res, p1, k1, s1)
        k2 = net_params['kernal_2']
        p2 = net_params['padding_2']
        s2 = net_params['stride_2']
        
        width3 = conv_width(width2, p2, k2, s2)
        
        pk1 = net_params['pool_kernal_1']
        pp1 = net_params['pool_padding_1']
        ps1 = net_params['pool_stride_1']
        
        pool_width1 = pool_width(width3, pp1, pk1, ps1)
        
        k3 = net_params['kernal_3']
        p3 = net_params['padding_3']
        s3 = net_params['stride_3']
        
        width4 = conv_width(width3, p3, k3, s3)
        
        k4 = net_params['kernal_4']
        p4 = net_params['padding_4']
        s4 = net_params['stride_4']
        
        width5 = conv_width(width4, p4, k4, s4)
        
        pk2 = net_params['pool_kernel_2']
        pp2 = net_params['pool_padding_2']
        ps2 = net_params['pool_stride_2']
        
        pool_width2 = pool_width(width5, pp2, pk2, ps2)
        
        k5 = net_params['kernal_5']
        p5 = net_params['padding_5']
        s5 = net_params['stride_5']
        
        width6 = conv_width(width5, p5, k5, s5)
        
        conv1 = net_params['conv1_out_channels']
        conv2 = net_params['conv2_out_channels']
        conv3 = net_params['conv3_out_channels']
        conv4 = net_params['conv4_out_channels']
        conv5 = net_params['conv5_out_channels']
        
        hidden1 = net_params['hidden_1']
        hidden2 = net_params['hidden_2']
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, conv1, k1, s1, p1),
            nn.ReLU())
        self.batch1 = nn.BatchNorm2d(conv1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1, conv2, k2, s2, p2),
            nn.ReLU())
        self.batch2 = nn.BatchNorm2d(conv2)
        self.pool1 = nn.MaxPool2d(pk1,ps1,pp1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(conv2, conv3, k3, s3, p3),
            nn.ReLU())
        self.batch3 = nn.BatchNorm2d(conv3)
        self.layer4 = nn.Sequential(
            nn.Conv2d(conv3, conv4, k4, s4, p4),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(pk2, ps2, pp2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(conv4, conv5, k5, s5, p5),
            nn.ReLU())
        self.batch4 = nn.BatchNorm2d(conv5)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear((width6**2)*conv5, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, data.num_emotions)
        
        
    def forward(self, x):
        x = self.layer1(x)
#        x = self.batch1(x)
        x = self.layer2(x)
#        x = self.batch2(x)
#        x = self.pool1(x)
        x = self.drop_out(x)
        x = self.layer3(x)
        x = self.batch3(x)
        x = self.layer4(x)
#        x = self.pool2(x)
        x = self.layer5(x)
        x = self.batch4(x)
        x = x.reshape(x.size(0),-1)
        x = self.drop_out(x)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        
        return func.softmax(x,dim=(1))
    
    def backprop(self, inputs, targets, loss, optimizer):
        self.train()
        
        inputs = inputs.float()
        targets = targets.float()
        
        outputs = self.forward(inputs)
        obj_val = loss(outputs, targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
    
    def test(self, data, loss):
        
        inputs = data.test_data
        targets = data.test_target
        
        self.eval()
        with torch.no_grad():
            inputs = inputs.float()
            targets = targets.float()
            
            cross_val = loss(self.forward(inputs), targets)
        
        return cross_val.item()
    



 

if __name__ == '__main__':
    
    data_path = os.getcwd() + '/archive'
    data = Data(data_path)