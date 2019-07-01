import numpy as np
import os
import json
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from processing import set_device, save_checkpoint, train_model, test_model, transform_data, load_datasets, load_json
from model_selection import select_model
from train_arg_parser import get_input_arguments

#################################

#################################

def main():
    
    parser = get_input_arguments()
    
    # check for data directory
    if not os.path.isdir(parser.data_directory):
        print(f'Cannot locate data directory: {parser.data_directory}, please enter another directory.')
        exit(1)
    
    #if not os.path.isdir(parser.validation_directory):
        #print(f'Cannot locate data directory: {parser.validation_directory}, please enter another directory.')

    # check for save directory
    if not os.path.isdir(parser.save_dir):
        print(f'Creating directory: {parser.save_dir}')
        os.makedirs(parser.save_dir)
    
    device = set_device(parser.use_gpu)
    
    # Map categories to their respective names
    cat_to_name = load_json(parser.category_json)
    #with open(parser.category_json, 'r') as f:
        #cat_to_name = json.load(f)
    
    output_size = len(cat_to_name)
    print(f'There are {output_size} categories in the dataset, meaning an output layer of {output_size} units')    
    
    '''data_dir = parser.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    '''
    
    train_transform, valid_transform, test_transform = transform_data()
    train_dataset, valid_dataset, test_dataset = load_datasets(parser.data_directory, train_transform, valid_transform, test_transform)
    #train_dataset, valide_dataset, test_dataset
    # takes data from train, validation and test set folders, performs transforms, loads sets with batch sizes
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    validationloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
                   
    # pass architecture and hidden units as arguments, returns the loaded architecture model, classifier, optimizer and NLLLoss function
    model, criterion, optimizer, classifier = select_model(parser.hidden_units, output_size, parser.learnrate, device, parser.arch)
    
    train_model(device, parser.epochs, trainloader, validationloader, model, optimizer, criterion)
        
    test_model(device, testloader, model)
        
    save_checkpoint(parser.save_dir, train_dataset, model, classifier, optimizer, parser.epochs, parser.arch)  

    
# Call to main function to run the program
if __name__ == "__main__":
    main()    
    
    
    
    