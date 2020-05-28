# -*- coding: utf-8 -*-
#
# source: https://gist.github.com/viniciusarruda/bea4a4ee06c2e9558d1b53083516a7e5
# Author:     Vinicius Ferraco Arruda                   
# Email:      viniciusferracoarruda@gmail.com            
# Website:    viniciusarruda.github.io                   
#

import argparse
from PIL import Image
import numpy as np
import os
import keras.backend as K
from keras.models import model_from_json
os.environ['KERAS_BACKEND']='tensorflow' 


parser = argparse.ArgumentParser(description='Generate fake images given a trained model.')
parser.add_argument('--model_dir', required=True, help='Path where the generators models are saved.')
#parser.add_argument('--input_a_dir', required=True, help='Path where images in domain A are.')
parser.add_argument('--input_b_dir', required=True, help='Path where images in domain B are.')
#parser.add_argument('--output_a2b_dir', required=True, help='Path where the fake B images will be saved.')
parser.add_argument('--output_b2a_dir', required=True, help='Path where the fake A images will be saved.')
args = parser.parse_args()


def load_model(filepath):
    # load json and create model
    with open('{}.json'.format(filepath), 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    print("model loaded, now weights")
    # load weights into new model
    loaded_model.load_weights("{}.h5".format(filepath))
    print("weights loaded")
    return loaded_model

def generate_function(netG):
    real_input = netG.inputs[0]
    fake_output = netG.outputs[0]  
    fn_generate = K.function([real_input], [fake_output])
    return fn_generate

def read_image(fn):
    loadSize = 128
    im = Image.open(fn).convert('RGB')
    im = im.resize( (loadSize, loadSize), Image.BILINEAR )
    img = np.array(im)/255*2-1
    return img[None, :]

def generate_imgs(input_dir, output_dir, generator):

    for img_file in os.listdir(input_dir):
        img = read_image(os.path.join(input_dir, img_file))
        fake_img = generator([img])[0][0]
        fake_img = Image.fromarray(((fake_img+1)/2*255).clip(0,255).astype('uint8'))
        fake_img.save(os.path.join(output_dir, img_file))


print("Before model netGB loaded")
netGA = load_model(args.model_dir + 'netGA')
#netGB = load_model(args.model_dir + 'netGB')
print("After model netGB loaded")

# netGA.summary()
# netGB.summary()

#generate_a2b = generate_function(netGB)
generate_b2a = generate_function(netGA)

#generate_imgs(args.input_a_dir, args.output_a2b_dir, generate_a2b)
generate_imgs(args.input_b_dir, args.output_b2a_dir, generate_b2a)