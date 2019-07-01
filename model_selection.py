import torch
from torchvision import models
from torch import nn, optim

# dictionary of necessary input sizes for architectures
pre_t_models = {'vgg16' : 25088,
                'alexnet' : 9216,
                'densenet121' : 1024}

def select_model(hidden_units, output_size, learnrate, device, architecture='vgg16'):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)   
        

    #freeze parameters of the pretrained model so they're not modified with gradient descent
    for parameters in model.parameters():
        parameters.requires_grad=False
    # set input size 
    input_size = pre_t_models[architecture]

    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.Dropout(0.4),
                               nn.ReLU(),
                               nn.Linear(hidden_units, output_size),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)

    model.to(device);

    return model, criterion, optimizer, classifier