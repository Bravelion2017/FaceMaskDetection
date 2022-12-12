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
preprocess_input2 = tf.keras.applications.resnet50.preprocess_input  #resnet50.preprocess_input
preprocess_input3 = tf.keras.applications.vgg16.preprocess_input #vgg16.preprocess_input


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
img_size = (299, 299)
img_size2 = (224, 224)
input_shape = (img_size, img_size, channel)

dataset = [parse_annotation(anno) for anno in glob.glob(annotations_path + '/*.xml')]
full_dataset = sum(dataset, [])
df = pd.DataFrame(full_dataset)
df.rename(columns = {'file':'file_name', 'name':'label'}, inplace = True)
df['file_name'] = [input_data_path + '/' + i + '.png' for i in df['file_name']]
original_df= df.copy(deep=True)
df['label'] = df['label'].map(lambda x: 'with_mask' if x=='mask_weared_incorrect' else x )

# Get labels
labels = df["label"].unique()


# Get train test split
# 20% Test | 80%* Train(20% Validation)
train_df, test_df = train_test_split(df, train_size=0.8, stratify=df['label'])
train_df, valid_df = train_test_split(train_df,train_size=0.8, stratify=train_df['label'])

# Making data in Tensorflow

test_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)
test_gen2 = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input2)
test_gen3 = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input3)

test_ds = test_gen.flow_from_dataframe(test_df, x_col='file_name', y_col='label',
                                         target_size=(299, 299), class_mode='sparse',
                                         batch_size=batch_size, shuffle=False )
test_ds2 = test_gen2.flow_from_dataframe(test_df, x_col='file_name', y_col='label',
                                         target_size=(224, 224), class_mode='sparse',
                                         batch_size=batch_size, shuffle=False )
test_ds3 = test_gen3.flow_from_dataframe(test_df, x_col='file_name', y_col='label',
                                         target_size=(224, 224), class_mode='sparse',
                                         batch_size=batch_size, shuffle=False )

# test_tensor = test_gen.flow_from_dataframe(test_imgs, x_col='images',
#                                          target_size=img_size, class_mode=None,
#                                          batch_size=batch_size, shuffle=False )

#====================================Pre-Trained Model Xception================================================
# The default input size for the pretrained model input_size = [299, 299]

# Add the pretrained layers
pretrained_model = keras.applications.Xception(include_top=False, weights='imagenet',input_shape=(299, 299, 3))
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)
output = keras.layers.Dense(n_classes, activation='sigmoid')(average_pooling) #initially: softmax
model = keras.Model(inputs=pretrained_model.input, outputs=output)

# Add the pretrained layers
pretrained_model2 = keras.applications.ResNet50(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
average_pooling2 = keras.layers.GlobalAveragePooling2D()(pretrained_model2.output)
output2 = keras.layers.Dense(n_classes, activation='sigmoid')(average_pooling2)
model2 = keras.Model(inputs=pretrained_model2.input, outputs=output2)

# Add the pretrained layers
pretrained_model3 = keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
average_pooling3 = keras.layers.GlobalAveragePooling2D()(pretrained_model3.output)
output3 = keras.layers.Dense(n_classes, activation='sigmoid')(average_pooling3)
model3 = keras.Model(inputs=pretrained_model3.input, outputs=output3)



#====================================Testing================================================

# Compile the models
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model3.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# labels
class_dict={'with_mask': 0, 'without_mask': 1}

# Load the saved models
model.load_weights(filepath=main_path + os.path.sep +'model_xception_facemask.h5')
model2.load_weights(filepath=main_path + os.path.sep +'model_resnet_facemask.h5')
model3.load_weights(filepath=main_path + os.path.sep +'model_vgg16_facemask.h5')

loss1, accuracy1= model.evaluate(test_ds)
loss2, accuracy2= model2.evaluate(test_ds2)
loss3, accuracy3= model3.evaluate(test_ds3)
# Comparing Models
lab= ['Xception','ResNet50', 'VGG16']
losses=[loss1, loss2, loss3]
acc= [accuracy1,accuracy2,accuracy3]
x = np.arange(len(lab))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, losses, width, label='Loss')
rects2 = ax.bar(x + width/2, acc, width, label='Accuracy')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_title('Comparing Models')
ax.set_xticks(x, lab)
ax.legend(loc='lower right')
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()

# Ensemble Models
pred1= model.predict(test_ds)
pred2= model2.predict(test_ds2)
pred3= model3.predict(test_ds3)
predictions1,predictions2,predictions3=[],[],[]
for i in range(len(pred1)):
    predictions1.append(np.argmax(pred1[i]))
    predictions2.append(np.argmax(pred2[i]))
    predictions3.append(np.argmax(pred3[i]))
rez= pd.DataFrame()
rez['res1']= np.array(predictions1)
rez['res2']= np.array(predictions2)
rez['res3']= np.array(predictions3)
rez_final= rez.mode(axis=1).values
check_df= pd.DataFrame(data=test_df.label.copy()).reset_index()
check_df['ensemble']= rez_final
check_df['ensemble']=check_df['ensemble'].map(lambda x:get_key(x, class_dict))

from sklearn.metrics import accuracy_score
ens_acc= accuracy_score(check_df.label,check_df.ensemble)
print(f"Ensemble Accuracy: {ens_acc}")


# My Test

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

model= keras.models.load_model('model_xception_facemask.h5') #You can use other pre-trained but set img_size = (size, size)

new_path= main_path + '/test/'
imgs=os.listdir(new_path)
test_imgs= imgs_to_df(path=new_path,images_list=imgs)
new_predictions = facemask(test_imgs,'images',test_gen,model,get_key,class_dict)
#Try Plots
labz= imgs
idx=1
for i in range(4):
    plt.subplot(2, 2, idx)
    pic = cv2.imread(test_imgs.images.values[i])
    plt.imshow(pic)
    plt.title(f"Label:{labz[i]}|Pred:{new_predictions[i]} ",fontdict={'size':10})
    idx +=1

plt.tight_layout()
plt.show()

# Photo Credit: Unsplash.com