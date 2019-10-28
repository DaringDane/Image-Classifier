# Image Classifier using Pytorch

As a project from the Udacity course, AI Programming with Python, I wrote this image classifier in Pytorch.
The model utilizes trasfer learning from one of three different chosen by the user or selects vgg16 by default.
Images are processed and sent through a classifier with a LogSoftmax output activation fed into a 
Negative Log Likelihood Loss (NLLL) function. 

## train.py - The Training Step

_To execute the training, see the [train_arg_parser](train_arg_parser.py) file for an example 
terminal command to enter, then modify listed parameters to set training to your preference._

Results with default parameters leads to ~91% accuracy during validation and test sets and prints updates as
the model trains.

Finally, a checkpoint.pth file is made which stores the trained model data for easy deployment for new 
image sets.

## predict.py - Use to make predictions on any picture (in line with category, like flowers)

_To execute the prediction, see the [predict_arg_parser](predict_arg_parser.py) file for an example 
terminal command to enter. Adjust the value of 'topk' to change how many top probable predictions are displayed._

* Accepts input of a novel PIL image and converts it to standardized size and data type
* Loads a checkpoint file with saved model data(weights and biases and other necessary 
parameters)
* The image is processed in the prediction function to return the top K classes predicted by the model

# Note:

The default image file in this repo used to train the model is flowers/ collected from [this database](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). You can download the
image dataset by clicking the link 'Dataset images' under downloads, then adding it to your local repository
titled 'flowers/', however another dataset can be uploaded and used in its stead by changing the terminal 
command to include the new image directory

**Small note: When I define my dictionary of models, I show 'VGG16', 'densenet121', and 'alexnet', but then mistakenly printed VGG19's model structure instead of VGG16. This doesn't hurt the model, as it still shows the same structure as VGG16 and is only used to show how it looks on the backend of the transfer learning model.**
