import json 
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', action="store", default = './test_images/orange_dahlia.jpg', type = str)
    parser.add_argument('--model_path', action="store", default = './my_model.h5', type = str)
    parser.add_argument("--top_k", action="store", default=5, type=int)
    parser.add_argument("--category_names", action="store",default="label_map.json")
    return parser.parse_args()

def process_img(img):
    image_res = 224
    image = np.squeeze(img)
    image = tf.image.resize(img, (image_res, image_res))/255.0
    return image

def predict():
    args = options()
    number_of_outputs = args.top_k
    json_name = args.category_names
    image_path = args.image_path
    model_path = args.model_path
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    with open(json_name, 'r') as f:
        label_to_name = json.load(f)

        
    img = Image.open(image_path)
    img = np.asarray(img)
    process_image = process_img(img) 
    predict = model.predict(np.expand_dims(process_image, axis=0))
    top_values, top_indices = tf.math.top_k(predict, number_of_outputs)
    top_classes = [label_to_name[str(value+1)] for value in top_indices.cpu().numpy()[0]]
    
    
    
    return top_values.numpy()[0], top_classes
    
if __name__=="__main__":
    probs, classes = predict()
    print("Top propabilities",probs)
    print('Top classes', classes)
    