import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# Function to build new classifier
def build_classifier(model, input_units, hidden_units, dropout):
    # Weights of pretrained model are frozen so we don't backprop through/update them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replacing the pretrained classifier with the one above
    model.classifier = classifier
    return model



def validation(model, validloader, criterion, gpu_mode):
    valid_loss = 0
    accuracy = 0
    
    if gpu_mode == True:
    # change model to work with cuda
        model.to('cuda')
    else:
        pass
    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validloader):
        
        if gpu_mode == True:
        # Change images and labels to work with cuda
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass
        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy



def train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode):
    #epochs = 3
    steps = 0
    print_every = 10

    if gpu_mode == True:
    # change to cuda
        model.to('cuda')
    else:
        pass
    
    for e in range(epochs):
        #since = time.time()
        running_loss = 0

        # Carrying out training step
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            if gpu_mode == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass

            # zeroing parameter gradients
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Carrying out validation step
            if steps % print_every == 0:
                # setting model to evaluation mode during validation
                model.eval()
                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu_mode)
                print(f"No. epochs: {e+1}, \
                Training Loss: {round(running_loss/print_every,3)} \
                Valid Loss: {round(valid_loss/len(validloader),3)} \
                Valid Accuracy: {round(float(accuracy/len(validloader)),3)}")

                running_loss = 0
                # Turning training back on
                model.train()
    
    return model, optimizer



def test_model(model, testloader, gpu_mode):
    correct = 0
    total = 0
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):
            
            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test accuracy of model for {total} images: {round(100 * correct / total,3)}%")


    
def save_model(model, train_data, optimizer, save_dir, epochs):
    # Saving: feature weights, new classifier, index-to-class mapping, optimiser state, and No. of epochs
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}

    return torch.save(checkpoint, save_dir)



def load_checkpoint(model, save_dir, gpu_mode):
    
    
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
            
    
   
def predict(processed_image, loaded_model, topk, gpu_mode):
    # Predict the class (or classes) of an image using a trained deep learning model.
  
    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    
    if gpu_mode == True:
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()
    
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(processed_image)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list

            