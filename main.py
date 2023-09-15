# -*- coding: utf-8 -*-
"""
    Emotion Classification
    ---------------------------
    @author: Andrew Francey
    @date: 9/12/2023
    ---------------------------
    main.py
"""

import time
import torch
import json, argparse
import numpy as np
import pylab as plt
import torch.nn as nn
import source as src
import pickle
import winsound
from random import randint


def prep(net_param):
    data = src.Data(args.data_path)
    print('Data is finished being loaded into a workable data type,\n' +\
          'Beginning Training of the model now.')
    model = src.Net(data, net_param)
    
    return data, model

def run(param, model, data):
    
    optimizer = torch.optim.SGD(model.parameters(), lr = param['learning_rate'],
                                momentum = param['momentum'])
    
    loss = nn.BCELoss(reduction = 'mean')
    
    loss_vals = []
    cross_vals = []
    
    num_epochs = int(param['num_epochs'])
    if args.batch:
        num_batchs = int(param['num_batchs'])
    
        batch_size = data.train_size//num_batchs
    
    for epoch in range(0, num_epochs):
#        print('\nepoch: {}/{}\n'.format(epoch+1, num_epochs))
        if args.batch:
            batches = [[],[]] # Stores the batches, data and targets before use
            
            ## Creates a list of indices of the unused images to avoid doubles
            unused_img = []
            for j in range(0,data.train_size):
                unused_img.append(j)
            
            ## Creates the batches 
            for batch in range(num_batchs):
                batch_data = torch.tensor(np.zeros((batch_size,1,
                                                    data.res,data.res)))
                batch_target = torch.tensor(np.zeros((batch_size,
                                                      data.num_emotions)))
                
                for i in range(batch_size):
                    if args.shuffle:
                        m = randint(0, len(unused_img)-1)
                        n = unused_img.pop(m) # Removes the image from the list of possible values
                    else:
                        n = batch*batch_size + i
                    
                    x, y = data.__getitem__(n)
                    
                    batch_data[i] = x
                    batch_target[i] = y
                
                batches[0].append(batch_data)
                batches[1].append(batch_target)
            
            ## Train the model on the batches
            for i in range(num_batchs):
#                print('    batch: {}/{}'.format(i+1, num_batchs))
                
                ## Extract the batches from the list of batches
                batch_data = batches[0][i]
                batch_target = batches[1][i]
                
                train_val = model.backprop(batch_data, batch_target, loss, 
                                           optimizer)
                loss_vals.append(train_val)
        
        else:
            train_val = model.backprop(data.train_data, data.train_target, loss, optimizer)
            loss_vals.append(train_val)
        
        test_val = model.test(data, loss)
        cross_vals.append(test_val)
        
        if (epoch + 1) % param['display_epochs'] == 0:
            print('Epoch [{}/{}] ({:.2f}%)'.format(epoch+1, num_epochs,\
                                               ((epoch + 1)/num_epochs)*100) + \
                  '\tTraining Lass: {:.4f}'.format(train_val) + \
                      '\tTest Loss: {:.4f}'.format(test_val))
            winsound.Beep(1000,220)
    
    print('Final training Loss: {:.4f}'.format(loss_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))
    
    return loss_vals, cross_vals
    


if __name__ == '__main__':
    
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Classification for emotions")
    parser.add_argument('--param', default='param.json', help='Json file for the training hyperparameters.')
    parser.add_argument('--data-path', default='archive', help='Location of the training and testing data.')
    parser.add_argument('--model-file', default='emotion_classifier.pkl', help='Name of the file the model will be saved as.')
    parser.add_argument('--batch', default=True, help='Use batching in training.')
    parser.add_argument('--shuffle', default=True, help='Shuffle the data in training.')
    args = parser.parse_args()
    
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    data, model = prep(param["net"])
    loss_vals, cross_vals = run(param["exec"],model,data)
    
    with open(args.model_file, 'wb') as f:
        pickle.dump(model, f)
    f.close()
    
    train_time = time.time()-start_time
    print('The model trained in {} mins and {} seconds'.format(train_time//60,
                                                               train_time-(train_time//60)*60))
    
    
    