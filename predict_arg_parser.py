import argparse


def predict_args():
    
    parser = argparse.ArgumentParser(description="Load a checkpoint for a trained network, select an image to get a prediction on, choose top 'k' predictions to display, and print results",
                                     usage="python predict.py flowers/test/17/image_03855.jpg checkpoint.pth --topk 5 --gpu",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('image_path', action='store', help='path to image to be predicted by nn') 
    
    parser.add_argument('checkpoint_path', action='store', help='reference path to saved checkpoint of model')
    
    parser.add_argument('--topk', type=int, action='store', dest='topk', help='choose to "k" predictions for the selected image')
    
    parser.add_argument('--category_to_name', type=str, action='store', dest='category_json', default='cat_to_name.json', help='Maps categories/classes to names from the data')
    
    parser.add_argument('--gpu', action='store_true', dest='use_gpu', default=False, help='run training on gpu or cpu')
    
    in_args = parser.parse_args()
    return in_args