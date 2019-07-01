import json
import torch
import numpy as np
from torchvision import models
from PIL import Image
from processing import set_device, load_checkpoint, load_json
from predict_arg_parser import predict_args

def main():
    parser = predict_args()
    
    cat_to_name = load_json(parser.category_json)
        
    device = set_device(parser.use_gpu)
        
    # Select the Image Path - used for methods - load_checkpoint, process_image, and predict
    image_path = parser.image_path
    model, architecture = load_checkpoint(parser.checkpoint_path)
    topk = parser.topk
    
    model.to(device)
    print(model)
    top_probs, top_classes = predict(image_path, model, topk)
    top_classes, top_probs = np.array(top_classes), np.array(top_probs)
    prediction = top_classes[0]
    probability = top_probs[0]
    
    print('Results: **************************************\n')
    print(f'Image input: {image_path}\n\n',
          f'Checkpoint loaded: {parser.checkpoint_path}\n\n',
          f'Architecture modeled: {architecture}\n\n\n',
          f'Model prediction: {cat_to_name[prediction]}\n\n',
          f'Prediction Confidence: {100 * probability:.2f} %')

def predict(image_path, model, topk=5):   
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).type(torch.cuda.FloatTensor)

    # create a batch size of 1 to put through our model
    img_tensor = img_tensor.unsqueeze_(0) #maybe add _ for 'in place'
    output = model(img_tensor)
    print(output)

    ps = torch.exp(output)
    top_probs, top_indices = ps.topk(topk)

    top_probabilities = top_probs.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    top_classes = [idx_to_class[index] for index in top_indices]
    #top_classes = top_classes.type(torch.FloatTensor)
    
    print(top_classes, type(top_classes))
    print(top_probabilities, type(top_probabilities))

    return top_probabilities, top_classes
    
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image_path)

    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((3000, 256))
    else:
        pil_image.thumbnail((256, 3000))

    # set coordinates for center crop (coordinate plane starts at 0,0 in upperleft corner)
    left = (pil_image.width-244)/2
    upper = (pil_image.height-244)/2
    right = left + 244
    lower = upper + 244
    pil_image = pil_image.crop((left, upper, right, lower))

    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2, 0, 1))

    return np_image


    
# Call to main function to run the program
if __name__ == "__main__":
    main()    