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
        
        k3 = net_params['kernal_3']
        p3 = net_params['padding_3']
        s3 = net_params['stride_3']
        
        width4 = conv_width(width3, p3, k3, s3)
        
        k4 = net_params['kernal_4']
        p4 = net_params['padding_4']
        s4 = net_params['stride_4']
        
        width5 = conv_width(width4, p4, k4, s4)
        
        conv1 = net_params['conv1_out_channels']
        conv2 = net_params['conv2_out_channels']
        conv3 = net_params['conv3_out_channels']
        conv4 = net_params['conv4_out_channels']
        
        hidden1 = net_params['hidden_1']
        hidden2 = net_params['hidden_2']
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, conv1, k1, s1, p1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1, conv2, k2, s2, p2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(conv2, conv3, k3, s3, p3),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(conv3, conv4, k4, s4, p4),
            nn.ReLU())
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear((width5**2)*conv4, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, data.num_emotions)
        
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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