import numpy as np

import json
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def set_device(user_gpu_selection):
    if user_gpu_selection:
        if torch.cuda.is_available():
            device = "cuda"
            print("Running training on gpu.")
        else:
            device = "cpu"
            print("Gpu not available on your device. Running training on cpu.")                
    else:
        device = "cpu"
        print("Running training on cpu.")
    return device

def save_checkpoint(save_dir, train_dataset, model, classifier, optimizer, epochs, arch):
    print('Saving checkpoint now.')
    
    # Saving parameters of model
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'architecture' : arch,
                  'classifier' : classifier,
                  'optimizer' : optimizer,
                  'epochs' : epochs,
                  'class_to_idx' : model.class_to_idx,
                  'state_dict' : model.state_dict()}
       #if parser.save_dir_name.split('.')[1]:
        #save_name = parser.save_dir_name.split('.')[0]
        
    # If user enters directory name with a .anything extension, removes it, and replaces with .pth
    savepoint = save_dir.split('.')
    print(savepoint)
    #if savepoint[1]:
        #savepoint = savepoint[0]
        
    save_dir = f'{save_dir}'
    torch.save(checkpoint, f'{save_dir}.pth')
    
    print(f'Checkpoint saved at directory: {save_dir}.pth')
    
def load_checkpoint(image_path):
    checkpoint = torch.load(image_path)
    arch_chosen = checkpoint['architecture']
    if arch_chosen == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch_chosen == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch_chosen == 'densenet121':                
        model = models.densenet121(pretrained=True)
    print(arch_chosen + " is the present pretrained model.\n")
    
    for param in model.parameters():
        param.requires_grad = False
    architecture = checkpoint['architecture']
    model.classifier = checkpoint['classifier']
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, architecture
    
def train_model(device, epochs, trainloader, validationloader, model, optimizer, criterion):
    running_loss = 0
    steps = 0
    check_every = 10
    accuracy = 0
    
    print("Beginning training sequence *play Eye of the Tiger while you wait*")
    print(f"Printing update every {check_every} weight steps.")
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # run images through the model, return log probabilities from log softmax function
            log_output = model(images)
            
            # calculate loss, implement gradient descent and apply back propogation, then apply to weights
            loss = criterion(log_output, labels)
            loss.backward()
            optimizer.step()
            
            # take count of growing loss quantity for the set
            running_loss += loss.item()
            
            if steps % check_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validationloader:
                        images, labels = images.to(device), labels.to(device)

                        logps = model(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        #convert log probabilities to standard probabilities from 0.0-1.0
                        ps = torch.exp(logps)
                        
                        #create a binary tensor, equal, with values 1=True and 0=False when comparing predicted class to label
                        equal = (labels.data == ps.max(dim=1)[1])
                        #convert equal to FloatTensor so torch can use it, then take mean of all correct values
                        accuracy += torch.mean(equal.type(torch.FloatTensor))


                print(f"Epoch {epoch+1}, Step {steps} of {len(trainloader)*epochs} "
                      f"Training Loss: {running_loss/check_every:.3f}  "
                      f"Validation Loss: {valid_loss/len(validationloader):.3f}  "
                      #f"Accuracy: {100 * correct/total:.2f} %")
                      f"Accuracy: {100 * accuracy/len(validationloader):.2f} %")
                running_loss = 0
                model.train()
    print("Training complete")
                
def test_model(device, testloader, model):
    accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            print(output[0])
            probabilities = torch.exp(output)
            #torch.max takes data from the output from columns(1) of classes and selects the max value, returning 
            # a tuple of (values, indeces). Output is a 2D tensor of (batchsize, 102)
            equal = (labels.data == probabilities.max(dim=1)[1])
            #convert equal to FloatTensor so torch can use it, then take mean of all correct values
            accuracy += torch.mean(equal.type(torch.FloatTensor))
        
    print(f'Accuracy of Network on Test Set: {accuracy/len(testloader) * 100:.2f} %')
    
def transform_data():
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    return train_transform, valid_transform, test_transform

def load_datasets(data_dir, train_transform, valid_transform, test_transform):
    train_dataset = datasets.ImageFolder(data_dir + 'train/', transform=train_transform)
    valid_dataset = datasets.ImageFolder(data_dir + 'valid/', transform=valid_transform)
    test_dataset = datasets.ImageFolder(data_dir + 'test/', transform=test_transform)
               
    return train_dataset, valid_dataset, test_dataset

def load_json(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
        
