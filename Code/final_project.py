import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score, classification_report
import cv2
import seaborn as sns
from tensorflow import keras
#
input_data_path = '/home/ubuntu/project/archive/images'
annotations_path = "/home/ubuntu/project/archive/annotations"
images = [*os.listdir("/home/ubuntu/project/archive/images")]
output_data_path =  '.'
#%%
IMG_SIZE = 100
batch_size = 8
epochs = 50
#%%
def parse_annotation(path):
    tree = ET.parse(path)
    root = tree.getroot()
    constants = {}
    objects = [child for child in root if child.tag == 'object']
    for element in tree.iter():
        if element.tag == 'filename':
            constants['file'] = element.text[0:-4]
        if element.tag == 'size':
            for dim in list(element):
                if dim.tag == 'width':
                    constants['width'] = int(dim.text)
                if dim.tag == 'height':
                    constants['height'] = int(dim.text)
                if dim.tag == 'depth':
                    constants['depth'] = int(dim.text)
    object_params = [parse_annotation_object(obj) for obj in objects]
    # print(constants)
    full_result = [merge(constants, ob) for ob in object_params]
    return full_result


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


def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res
#%%
dataset = [parse_annotation(anno) for anno in glob.glob(annotations_path+"/*.xml") ]

full_dataset = sum(dataset, [])

df = pd.DataFrame(full_dataset)
df.shape
#%%
print(df.head().to_string())
#%%
final_test_image = 'maksssksksss0'
df_final_test = df.loc[df["file"] == final_test_image]
images.remove(f'{final_test_image}.png')
df = df.loc[df["file"] != final_test_image]
#%%
df["name"].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Name')
plt.xticks(rotation = 0)
plt.tight_layout()
plt.show()
#%%
labels = df['name'].unique()
directory = ['train', 'test', 'val']
output_data_path = '.'

import os
for label in labels:
    for d in directory:
        path = os.path.join(output_data_path, d, label)
        if not os.path.exists(path):
            os.makedirs(path)
#%%
def crop_img(image_path, x_min, y_min, x_max, y_max):
    x_shift = (x_max - x_min) * 0.1
    y_shift = (y_max - y_min) * 0.1
    img = Image.open(image_path)
    cropped = img.crop((x_min - x_shift, y_min - y_shift, x_max + x_shift, y_max + y_shift))
    return cropped
#%%
def extract_faces(image_name, image_info):
    faces = []
    df_one_img = image_info[image_info['file'] == image_name[:-4]][['xmin', 'ymin', 'xmax', 'ymax', 'name']]
    for row_num in range(len(df_one_img)):
        x_min, y_min, x_max, y_max, label = df_one_img.iloc[row_num]
        image_path = os.path.join(input_data_path, image_name)
        faces.append((crop_img(image_path, x_min, y_min, x_max, y_max), label,f'{image_name[:-4]}_{(x_min, y_min)}'))
    return faces
#%%
cropped_faces = [extract_faces(img, df) for img in images]
#%%
flat_cropped_faces = sum(cropped_faces, [])
#%%
with_mask = [(img, image_name) for img, label,image_name in flat_cropped_faces if label == "with_mask"]
mask_weared_incorrect = [(img, image_name) for img, label,image_name in flat_cropped_faces if label == "mask_weared_incorrect"]
without_mask = [(img, image_name) for img, label,image_name in flat_cropped_faces if label == "without_mask"]
#%%
print(len(with_mask))
print(len(without_mask))
print(len(mask_weared_incorrect))
print(len(with_mask) + len(without_mask) + len(mask_weared_incorrect))
#%%
train_with_mask, test_with_mask = train_test_split(with_mask, test_size=0.20, random_state=42)
test_with_mask, val_with_mask = train_test_split(test_with_mask, test_size=0.7, random_state=42)

train_mask_weared_incorrect, test_mask_weared_incorrect = train_test_split(mask_weared_incorrect, test_size=0.20, random_state=42)
test_mask_weared_incorrect, val_mask_weared_incorrect = train_test_split(test_mask_weared_incorrect, test_size=0.7, random_state=42)

train_without_mask, test_without_mask = train_test_split(without_mask, test_size=0.20, random_state=42)
test_without_mask, val_without_mask = train_test_split(test_without_mask, test_size=0.7, random_state=42)
#%%
def save_image(image, image_name, output_data_path,  dataset_type, label):
    output_path = os.path.join(output_data_path, dataset_type, label ,f'{image_name}.png')
    image.save(output_path)
#%%
for image, image_name in train_with_mask:
    save_image(image, image_name, output_data_path, 'train', 'with_mask')

for image, image_name in train_mask_weared_incorrect:
    save_image(image, image_name, output_data_path, 'train', 'mask_weared_incorrect')

for image, image_name in train_without_mask:
    save_image(image, image_name, output_data_path, 'train', 'without_mask')

for image, image_name in test_with_mask:
    save_image(image, image_name, output_data_path, 'test', 'with_mask')

for image, image_name in test_mask_weared_incorrect:
    save_image(image, image_name, output_data_path, 'test', 'mask_weared_incorrect')

for image, image_name in test_without_mask:
    save_image(image, image_name, output_data_path, 'test', 'without_mask')

for image, image_name in val_with_mask:
    save_image(image, image_name, output_data_path, 'val', 'with_mask')

for image, image_name in val_without_mask:
    save_image(image, image_name, output_data_path, 'val', 'without_mask')

for image, image_name in val_mask_weared_incorrect:
    save_image(image, image_name, output_data_path, 'val', 'mask_weared_incorrect')
#%%
model = Sequential()
model.add(Conv2D(16, 3,  padding='same', activation = 'relu', input_shape = (IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(2))
model.add(Conv2D(32, 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(256, 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units = 2304, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 3, activation = 'softmax'))

#%%
# Train, evaluate and save the best model
model.summary()
#%%
datagen = ImageDataGenerator(
    rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
    height_shift_range=0.1, rotation_range=4, vertical_flip=False)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)
train_generator = datagen.flow_from_directory(
    directory='/home/ubuntu/project/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical", batch_size=batch_size, shuffle=True
)
# Validation data
val_generator = val_datagen.flow_from_directory(
    directory='/home/ubuntu/project/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical", batch_size=batch_size, shuffle=True
)
# Test data
test_generator = val_datagen.flow_from_directory(
    directory='/home/ubuntu/project/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical", batch_size=batch_size, shuffle=False
)
#%%
data_size = len(train_generator)
steps_per_epoch = int(data_size / batch_size)
print(f"steps_per_epoch: {steps_per_epoch}")

val_steps = int(len(val_generator) // batch_size)
print(f"val_steps: {val_steps}")
#%%
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuracy', 'Recall', 'Precision', 'AUC']
)
#%%
model_checkpoint_cb = keras.callbacks.ModelCheckpoint('/home/ubuntu/project/archive/model.h5',
                                                      save_best_only=True,
                                                      save_weights_only=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
#%%
lrr = ReduceLROnPlateau(monitor='val_loss',patience=8,verbose=1,factor=0.5, min_lr=0.00001)
#%%
model_history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    shuffle=True,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[early_stopping, lrr,model_checkpoint_cb]
)
#%%
predictions = model.predict(test_generator)
predictions
#%%
his = pd.DataFrame(model_history.history)
plt.plot(his["loss"],label = "Loss")
plt.plot(his["accuracy"],label = "accuracy")
plt.plot(his["lr"],label = "Learning rate")
plt.legend()
plt.title("History of the Model")
plt.xlabel("epochs")
plt.ylabel("value")
plt.show()
#%%
paths = test_generator.filenames
y_pred = model.predict(test_generator).argmax(axis=1)
classes = test_generator.class_indices

a_img_rand = np.random.randint(0,len(paths))
img = cv2.imread(os.path.join(output_data_path,'test', paths[a_img_rand]))
colored_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

plt.imshow(colored_img)
true_label = paths[a_img_rand].split('/')[0]
predicted_label = list(classes)[y_pred[a_img_rand]]
print(f'{predicted_label} || {true_label}')
#%%
def evaluation(y, y_hat, title='Confusion Matrix'):
    cm = confusion_matrix(y, y_hat,normalize="true")
    sns.heatmap(cm, cmap='PuBu', annot=True, fmt='g', annot_kws={'size': 20})
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(title, fontsize=18)

    plt.show()
#%%
y_true = test_generator.labels
y_pred = model.predict(test_generator).argmax(axis=1) # Predict prob and get Class Indices
#%%
evaluation(y_true, y_pred)
#%%
print("Test accuracy:",accuracy_score(y_true,y_pred))
#%%
print(classes)
#%%
def getCalssName(classNo):
  if   classNo == 0: return 'mask_worn_incorrect'
  elif classNo == 1: return 'with_mask'
  elif classNo == 2: return 'without_mask'
#%%
cur_Img1 = cv2.imread("/home/ubuntu/project/archive/withmask2.jpg")
cur_Img1 = cv2.cvtColor(cur_Img1, cv2.COLOR_RGB2BGR)

cur_Img = np.array(cur_Img1)
cur_Img = cv2.resize(np.float32(cur_Img), (IMG_SIZE, IMG_SIZE))
cur_Img = cur_Img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

predictions = model.predict(cur_Img)
plt.imshow(cur_Img1)
plt.title(f"{getCalssName(predictions.argmax(axis=1))}")
plt.show()
#%%
cur_Img1 = cv2.imread("/home/ubuntu/project/archive/nomask21.jpeg")
cur_Img1 = cv2.cvtColor(cur_Img1, cv2.COLOR_RGB2BGR)

cur_Img = np.array(cur_Img1)
cur_Img = cv2.resize(np.float32(cur_Img), (IMG_SIZE, IMG_SIZE))
cur_Img = cur_Img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

predictions = model.predict(cur_Img)
plt.imshow(cur_Img1)
plt.title(f"{getCalssName(predictions.argmax(axis=1))}")
plt.show()
#%%
