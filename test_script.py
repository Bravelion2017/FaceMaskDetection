_="""
    FACE MASK DETECTION USING TENSORFLOW
        By: Atharva and Osemekhian
"""
print(_)

#====================================Libraries imports================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, os, shutil, cv2, glob, math
from tqdm import tqdm
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from tensorflow import keras
import cv2
import os

batch_size = 16
epochs = 30
img_size= 299
channel= 3
n_classes= 2
LR= 1e-4
random_state= 42
main_path= os.getcwd()

#====================================Helper Functions================================================

input_data_path = '/home/ubuntu/finalproject'+'/images'
annotations_path ='/home/ubuntu/finalproject'+ "/annotations"

# Set the preprocess_input of the pretrained model
preprocess_input = tf.keras.applications.xception.preprocess_input  #xception.preprocess_input


def get_key(value, dictionary):
  for key, val in dictionary.items():
    if value == val:
      return key

def resize(data, label):
    """
    Resize the data into the default input size of the pretrained model

    Parameters
    ----------
    data: the data
    label: the label

    Returns
    ----------
    The resized data and label
    """

    # Resize the data into the default input size of the pretrained model
    data_resized = tf.image.resize(data, input_size)

    return data_resized, label

def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res



def parse_annotation_object(annotation_object):
    params = {}
    for param in list(annotation_object):
        if param.tag == 'name':
            params['name'] = param.text
        if param.tag == 'bndbox':
            for coord in list(param):
                if coord.tag == 'xmin':
                    params['xmin'] = int(coord.text)
                if coord.tag == 'ymin':
                    params['ymin'] = int(coord.text)
                if coord.tag == 'xmax':
                    params['xmax'] = int(coord.text)
                if coord.tag == 'ymax':
                    params['ymax'] = int(coord.text)
    return params


def parse_annotation(path):
    tree = ET.parse(path)
    root = tree.getroot()
    constants = {}
    objects = [child for child in root if child.tag == 'object']
    for element in tree.iter():
        if element.tag == 'filename':
            constants['file'] = element.text[0: -4]
        if element.tag == 'size':
            for dim in list(element):
                if dim.tag == 'width':
                    constants['width'] = int(dim.text)
                if dim.tag == 'height':
                    constants['height'] = int(dim.text)
                if dim.tag == 'depth':
                    constants['depth'] = int(dim.text)
    object_params = [parse_annotation_object(obj) for obj in objects]
    full_result = [merge(constants, ob) for ob in object_params]
    return full_result

#====================================Data Generation================================================

# Image size
img_size = (img_size, img_size) #(35, 35)
input_shape = (img_size, img_size, channel)

# Making data in Tensorflow

test_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)

# test_tensor = test_gen.flow_from_dataframe(test_imgs, x_col='images',
#                                          target_size=img_size, class_mode=None,
#                                          batch_size=batch_size, shuffle=False )

#====================================Pre-Trained Model Xception================================================
# The default input size for the pretrained model input_size = [299, 299]

# Add the pretrained layers
pretrained_model = keras.applications.Xception(include_top=False, weights='imagenet',input_shape=(299, 299, 3))

# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

# Add the output layer
output = keras.layers.Dense(n_classes, activation='sigmoid')(average_pooling) #initially: softmax

# Get the model
model = keras.Model(inputs=pretrained_model.input, outputs=output)

print(model.summary())




#====================================Testing================================================

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# labels
class_dict={'with_mask': 0, 'without_mask': 1}

# Load the saved model
model.load_weights(filepath=main_path + os.path.sep +'model_xception_facemask.h5')

# check_test.pred= check_test.pred.map(lambda x:get_key(x,class_dict) )



# My Test
new_path= main_path + '/test/'
imgs=os.listdir(new_path)
dat=[]
for i in range(len(imgs)):
    dat.append(new_path+imgs[i])
test_imgs= pd.DataFrame(data=dat, columns=['images'])

def imgs_to_df(path, images_list):
    dat = []
    for i in range(len(images_list)):
        dat.append(path + images_list[i])
    test_imgs = pd.DataFrame(data=dat, columns=['images'])

    return test_imgs


def facemask(image_df,column_name, image_generator,model,get_key, class_dict, batch_size=16):
    ''':image_df : Should be in Dataframe
    :column_name : name of image column in string
    :image_generator : Tensorflow image generator with preprocessing function
    :model : saved model
    :get_key : function to check np.argmax value then return key
    :class_dict : dictionary for classes
    Image size is (299,299) for Xception Model
    '''
    tensor= image_generator.flow_from_dataframe(image_df, x_col=column_name,
                                         target_size=img_size, class_mode=None,
                                         batch_size=batch_size, shuffle=False)
    new_pred = model.predict(tensor)
    prediction = []
    for i in range(len(new_pred)):
        prediction.append(np.argmax(new_pred[i]))
    pred_df = pd.DataFrame(prediction, columns=['label'])
    pred_df.label = pred_df.label.map(lambda x: get_key(x, class_dict))

    return pred_df.label

new_predictions = facemask(test_imgs,'images',test_gen,model,get_key,class_dict)

#==Test 2
path2=os.getcwd()+os.path.sep+'test2/'
images_test= imgs_to_df(path=path2,images_list=os.listdir(path2))
new_predictions2 = facemask(images_test,'images',test_gen,model,get_key,class_dict)