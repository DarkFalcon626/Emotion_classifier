# -*- coding: utf-8 -*-
"""
    Emotion Classification
    ------------------------------
    @author: Andrew Francey
    @date: 9/12/2023
    ------------------------------
    Emotion identifier
"""

import pickle
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

class Data():
    
    def __init__(self, pic_location):
        
        image = Image.open(pic_location)
        image_array = torch.tensor(np.asarray(image))
        
        self.res = image.size[0]
        self.image = image_array
        
def identify(model,pic_location):
    
    ## Convert picture to wokring data tensor
        
    pic = Data(pic_location)
    
    result = model.forward(pic)
    
#    emotions = model.emotion_lst
    emotions = ['angry','happy','sad']
    
    i = torch.argmax(result)
    
    print('image {} shows the emotion {}'.format(pic_location,emotions[i]))
    return emotions[i]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Identification of the emotion in a picture")
    parser.add_argument('--pic_location', default="archive/test/happy/PrivateTest_1140198.jpg",
                        help="Location of the image to identify")
    parser.add_argument('--model', default="emotion_classifier.pkl",
                        help="Location of the model to for classification.")
    
    args = parser.parse_args()
    
    with open(args.model,'rb') as f:
        model = pickle.load(f)
    f.close()
    
    identify(model, args.pic_location)