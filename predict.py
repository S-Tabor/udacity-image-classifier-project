import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

from util_functions import load_data, process_image
from functions import load_checkpoint, predict, test_model


parser = argparse.ArgumentParser(description='Use neural network to make prediction on image.')

parser.add_argument('--image_path', action='store',
                    default = '../aipnd-project/flowers/test/102/image_08004',
                    help='Enter path to image.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'checkpoint.pth',
                    help='Enter location to save checkpoint in.')

parser.add_argument('--arch', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='Enter pretrained model to use, default is VGG-11.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 3,
                    help='Enter number of top most likely classes to view, default is 3.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter path to image.')

parser.add_argument('--gpu', action="store_true", default=False,
                    help='Turn GPU mode on or off, default is off.')

results = parser.parse_args()

save_dir = results.save_directory
image = results.image_path
top_k = results.topk
gpu_mode = results.gpu
cat_names = results.cat_name_dir
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Establish model template
pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

# Load model
loaded_model = load_checkpoint(model, save_dir, gpu_mode)

# Preprocess image - assumes jpeg format
processed_image = process_image(image)

if gpu_mode == True:
    processed_image = processed_image.to('cuda')
else:
    pass

# Carry out prediction
probs, classes = predict(processed_image, loaded_model, top_k, gpu_mode)

# Print probabilities and predicted classes
print(probs)
print(classes) 

names = []
for i in classes:
    names += [cat_to_name[i]]
    
# Print name of predicted flower with highest probability
print(f"This flower is most likely to be a: '{names[0]}' with a probability of {round(probs[0]*100,4)}% ")

