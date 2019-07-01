import argparse

'''
Example commands for the command line:
    - Select directory to save checkpoints in: python train.py data_directory --save_dir save_directory
    - Select training architecture: python train.py data_directory --arch "densenet121"
    - Set hyperparameters: python train.py data_directory --learning_rate 0.005 --hidden_units 2048 --epochs 8
    - Use GPU for training: python train.py data_directory --gpu

'''

def get_input_arguments():    
    parser = argparse.ArgumentParser(description='Set hyperparameters, architecture, train and validation datasets, and save state of the trained image classifier',
                                     usage="python train.py flowers/ --gpu --arch densenet121 --learnrate 0.001 --hidden_units 4096 --epochs 5",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_directory', type=str, action='store', help='Directory to train on')
    parser.add_argument('--category_to_name', type=str, action='store', dest='category_json', default='cat_to_name.json', help='Maps categories/classes to names from the data')
    
    parser.add_argument('--arch', type=str, dest='arch', default='vgg16', help='Pretrained model architecture options include: vgg16, densenet121, alexnet')
    # store the checkpoint in a directory
    parser.add_argument('--save_dir', action='store', dest='save_dir', default='checkpoint', type=str, help='Saves checkpoint in the current directory, or write directory to save into')
    
    #parser.add_argument('--save_dir_name', action='store', dest='save_dir_name', default='checkpoint', type=str, help='Choose name for save point directory')
    
    # Set command access to setting Hyperparameters for training with defaults of lr=0.002, Hiddenlayer=4096 units, epochs=5
    hyperparameters = parser.add_argument_group('hyperparameters')
    
    hyperparameters.add_argument('--learnrate', type=float, action='store', default='0.002', help='Learning rate')
    hyperparameters.add_argument('--hidden_units', type=int, action='store', default=4096, help='Units in hidden layer')
    hyperparameters.add_argument('--epochs', type=int, action='store', default=5, help='Number of epochs')
    
    # activate gpu processing
    parser.add_argument('--gpu', action='store_true', dest='use_gpu', default=False, help='run training on gpu or cpu')

    in_args = parser.parse_args()

    return in_args
