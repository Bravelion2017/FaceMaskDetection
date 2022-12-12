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

dataset = [parse_annotation(anno) for anno in glob.glob(annotations_path + '/*.xml')]
full_dataset = sum(dataset, [])
df = pd.DataFrame(full_dataset)
df.rename(columns = {'file':'file_name', 'name':'label'}, inplace = True)
df['file_name'] = [input_data_path + '/' + i + '.png' for i in df['file_name']]
original_df= df.copy(deep=True)
df['label'] = df['label'].map(lambda x: 'with_mask' if x=='mask_weared_incorrect' else x )
print(df.head())

#Encoding Target 'With Mask': 2,'Mask Worn Incorrectly': 1, 'Without Mask': 0
# df['label'] = df['label'].map(lambda x:2 if x=='with_mask' else (1 if x=='mask_weared_incorrect' else 0))

# Turn Target to Categorical type with:
# without_mask-TOP PRIORITY followed by
# 'mask_weared_incorrect'and 'with_mask'
# df['label'] = pd.Categorical(
#     df['label'], categories=['without_mask', 'mask_weared_incorrect', 'with_mask'], ordered=False
# )

# Get labels
labels = df["label"].unique()
print(labels)

# Get train test split
# 20% Test | 80%* Train(20% Validation)
train_df, test_df = train_test_split(df, train_size=0.8, stratify=df['label'])
train_df, valid_df = train_test_split(train_df,train_size=0.8, stratify=train_df['label'])
print(f"Train length:{len(train_df)} \nValidation length:{len(valid_df)} \n"
      f"Test length:{len(test_df)}")

# Image size
img_size = (img_size, img_size) #(35, 35)
input_shape = (img_size, img_size, channel)

# Making data in Tensorflow
train_gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
                               height_shift_range=0.1, rotation_range=4, vertical_flip=False, preprocessing_function=preprocess_input)
valid_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_dataframe(train_df, x_col='file_name', y_col='label',
                                         target_size=img_size, class_mode="sparse", #class_mode='sparse'
                                         batch_size=batch_size, shuffle=True,
                                         subset='training')

valid_ds = valid_gen.flow_from_dataframe(valid_df, x_col='file_name', y_col='label',
                                         target_size=img_size, class_mode='sparse',
                                         batch_size=batch_size, shuffle=True)

test_ds = test_gen.flow_from_dataframe(test_df, x_col='file_name', y_col='label',
                                         target_size=img_size, class_mode='sparse',
                                         batch_size=batch_size, shuffle=False )

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

#
# For each layer in the pretrained model
for layer in pretrained_model.layers:
    # Freeze the layer
    layer.trainable = False


#====================================CallBacks================================================
# -Using Performance scheduling to tune the learning rate
# -Using early stopping to handle overfitting
# ModelCheckpoint callback to_save_the_model

model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=main_path + '/model_xception_facemask.h5',
                                                      save_best_only=True,
                                                      save_weights_only=True)
# EarlyStopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

# ReduceLROnPlateau callback #Performance Scheduling
reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                            patience=2)

#====================================Compile Model================================================
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),# or Adam
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#====================================Initial Training================================================
# Train, evaluate and save the best model
history = model.fit(train_ds,
                    epochs=2,
                    validation_data=valid_ds,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.legend(loc='center')
plt.title("Learning Curve")
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# Unfreezing the  Pre-Trained layers (132)
for layer in pretrained_model.layers:
    # Unfreeze the layer
    layer.trainable = True

# Here we use a lower learning rate (by a factor of 10) of Adam optimizer,
# so that it is less likely to compromise the pretrained weights.

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model.fit(train_ds,
                    epochs=10,
                    validation_data=valid_ds,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.legend(loc='center')
plt.title("Learning Curve")
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()



#=========================================Pre-Trained Model 2===========================================================
# Set the preprocess_input of the pretrained model
preprocess_input2 = tf.keras.applications.resnet50.preprocess_input  #resnet50.preprocess_input
img_size2 = (224, 224)
# Making data in Tensorflow
train_gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
                               height_shift_range=0.1, rotation_range=4, vertical_flip=False, preprocessing_function=preprocess_input2)
valid_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input2)
test_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input2)

train_ds = train_gen.flow_from_dataframe(train_df, x_col='file_name', y_col='label',
                                         target_size=img_size2, class_mode="sparse", #class_mode='sparse'
                                         batch_size=batch_size, shuffle=True,
                                         subset='training')

valid_ds = valid_gen.flow_from_dataframe(valid_df, x_col='file_name', y_col='label',
                                         target_size=img_size2, class_mode='sparse',
                                         batch_size=batch_size, shuffle=True)

test_ds = test_gen.flow_from_dataframe(test_df, x_col='file_name', y_col='label',
                                         target_size=img_size2, class_mode='sparse',
                                         batch_size=batch_size, shuffle=False )

#==================================== Pre-Trained Model ResNet50 ================================================
# The default input size for the pretrained model input_size = [224, 224]

# Add the pretrained layers
pretrained_model2 = keras.applications.ResNet50(include_top=False, weights='imagenet',input_shape=(224, 224, 3))

# Add GlobalAveragePooling2D layer
average_pooling2 = keras.layers.GlobalAveragePooling2D()(pretrained_model2.output)

# Add the output layer
output2 = keras.layers.Dense(n_classes, activation='sigmoid')(average_pooling2)

# Get the model
model2 = keras.Model(inputs=pretrained_model2.input, outputs=output2)

print(model2.summary())

# For each layer in the pretrained model
for layer in pretrained_model2.layers:
    # Freeze the layer
    layer.trainable = False

#====================================CallBacks================================================
# -Using Performance scheduling to tune the learning rate
# -Using early stopping to handle overfitting
# ModelCheckpoint callback to_save_the_model

model_checkpoint_cb2 = keras.callbacks.ModelCheckpoint(filepath=main_path + '/model_resnet_facemask.h5',
                                                      save_best_only=True,
                                                      save_weights_only=True)
# EarlyStopping callback
early_stopping_cb2 = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

# ReduceLROnPlateau callback #Performance Scheduling
reduce_lr_on_plateau_cb2 = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                            patience=2)

#====================================Compile Model================================================
model2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),# or Adam
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#====================================Initial Training================================================
# Train, evaluate and save the best model
history = model2.fit(train_ds,
                    epochs=2,
                    validation_data=valid_ds,
                    callbacks=[model_checkpoint_cb2,
                               early_stopping_cb2,
                               reduce_lr_on_plateau_cb2])
# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.legend(loc='center')
plt.title("Learning Curve")
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# Unfreezing the  Pre-Trained layers (175)
for layer in pretrained_model2.layers:
    # Unfreeze the layer
    layer.trainable = True

# Here we use a lower learning rate (by a factor of 10) of Adam optimizer,
# so that it is less likely to compromise the pretrained weights.

# Compile the model
model2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model2.fit(train_ds,
                    epochs=10,
                    validation_data=valid_ds,
                    callbacks=[model_checkpoint_cb2,
                               early_stopping_cb2,
                               reduce_lr_on_plateau_cb2])

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.legend(loc='center')
plt.title("Learning Curve")
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()


#==================================== Pre-Trained Model VGG ================================================
# The default input size for the pretrained model input_size = [224, 224]
# Set the preprocess_input of the pretrained model
preprocess_input3 = tf.keras.applications.vgg16.preprocess_input #vgg16.preprocess_input
img_size2 = (224, 224)
# Making data in Tensorflow
train_gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
                               height_shift_range=0.1, rotation_range=4, vertical_flip=False, preprocessing_function=preprocess_input3)
valid_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input3)
test_gen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input3)

train_ds = train_gen.flow_from_dataframe(train_df, x_col='file_name', y_col='label',
                                         target_size=img_size2, class_mode="sparse", #class_mode='sparse'
                                         batch_size=batch_size, shuffle=True,
                                         subset='training')

valid_ds = valid_gen.flow_from_dataframe(valid_df, x_col='file_name', y_col='label',
                                         target_size=img_size2, class_mode='sparse',
                                         batch_size=batch_size, shuffle=True)

test_ds = test_gen.flow_from_dataframe(test_df, x_col='file_name', y_col='label',
                                         target_size=img_size2, class_mode='sparse',
                                         batch_size=batch_size, shuffle=False )



# Add the pretrained layers
pretrained_model3 = keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=(224, 224, 3))

# Add GlobalAveragePooling2D layer
average_pooling3 = keras.layers.GlobalAveragePooling2D()(pretrained_model3.output)

# Add the output layer
output3 = keras.layers.Dense(n_classes, activation='sigmoid')(average_pooling3)

# Get the model
model3 = keras.Model(inputs=pretrained_model3.input, outputs=output3)

print(model3.summary())

# For each layer in the pretrained model
for layer in pretrained_model3.layers:
    # Freeze the layer
    layer.trainable = False

#====================================CallBacks================================================
# -Using Performance scheduling to tune the learning rate
# -Using early stopping to handle overfitting
# ModelCheckpoint callback to_save_the_model

model_checkpoint_cb3 = keras.callbacks.ModelCheckpoint(filepath=main_path + '/model_vgg16_facemask.h5',
                                                      save_best_only=True,
                                                      save_weights_only=True)
# EarlyStopping callback
early_stopping_cb3 = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

# ReduceLROnPlateau callback #Performance Scheduling
reduce_lr_on_plateau_cb3 = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                            patience=2)

#====================================Compile Model================================================
model3.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),# or Adam
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#====================================Initial Training================================================
# Train, evaluate and save the best model
history = model3.fit(train_ds,
                    epochs=2,
                    validation_data=valid_ds,
                    callbacks=[model_checkpoint_cb3,
                               early_stopping_cb3,
                               reduce_lr_on_plateau_cb3])
# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.legend(loc='center')
plt.title("Learning Curve")
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# Unfreezing the  Pre-Trained layers (175)
for layer in pretrained_model3.layers:
    # Unfreeze the layer
    layer.trainable = True

# Here we use a lower learning rate (by a factor of 10) of Adam optimizer,
# so that it is less likely to compromise the pretrained weights.

# Compile the model
model3.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model3.fit(train_ds,
                    epochs=10,
                    validation_data=valid_ds,
                    callbacks=[model_checkpoint_cb3,
                               early_stopping_cb3,
                               reduce_lr_on_plateau_cb3])

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.legend(loc='center')
plt.title("Learning Curve")
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

#===================================================
# Testing:
class_dict=test_ds.class_indices

# Load the saved model
model.load_weights(filepath=main_path + os.path.sep +'model_xception_facemask.h5')
model2.load_weights(filepath=main_path + os.path.sep +'model_resnet_facemask.h5')
model3.load_weights(filepath=main_path + os.path.sep +'model_vgg16_facemask.h5')
loss, accuracy= model3.evaluate(test_ds)
pred= model.predict(test_ds)





# My Test
new_path= main_path + '/test/'
imgs=os.listdir(new_path)
dat=[]
for i in range(len(imgs)):
    dat.append(new_path+imgs[i])
test_imgs= pd.DataFrame(data=dat, columns=['images'])


test_tensor = test_gen.flow_from_dataframe(test_imgs, x_col='images',
                                         target_size=img_size, class_mode=None,
                                         batch_size=batch_size, shuffle=False )
new_pred= model3.predict(test_tensor)
prediction=[]
for i in range(len(new_pred)):
    prediction.append(np.argmax(new_pred[i]))
pred_df= pd.DataFrame(prediction,columns=['label'])
pred_df.label = pred_df.label.map(lambda x:get_key(x,class_dict))
print(pred_df)

# predictions=[]
# for i in range(len(pred)):
#     predictions.append(np.argmax(pred[i]))
# check_test= test_df.copy()
# check_test['pred']=predictions
# check_test.pred= check_test.pred.map(lambda x:get_key(x,class_dict) )
# print(check_test[['label','pred']].head(10))
