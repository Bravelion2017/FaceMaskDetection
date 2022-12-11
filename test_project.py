#%%
import numpy as np
from PIL import Image
from keras import models
import cv2
import os
import matplotlib.pyplot as plt

IMG_SIZE = 35
#%%
#Load the saved model

# #%%
# name_of_image = "maksssksksss2.png"
# curImg = cv2.imread(os.path.join("/home/ubuntu/project/archive","images",name_of_image))
# curImg = np.array(curImg)
#
# curImg = cv2.resize(np.float32(curImg), (IMG_SIZE,IMG_SIZE))
# curImg = curImg.reshape(-1,IMG_SIZE,IMG_SIZE,3)
# #%%
# predictions = model.predict(curImg)
# print(getCalssName(predictions[0]))

#%%
import dash
import dash_html_components as html
import numpy as np
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import math
from scipy.fft import fft
from scipy.special import expit
#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash("My App")
#%%
my_app.layout = html.Div([html.H5("Machine Learning 2 Final Project"),
                          dcc.Upload(
                              id='upload-image',
                              children=html.Div([
                                  'Drag and Drop or ',
                                  html.A('Select Files')
                                  ])),
                              html.Div(id = "output1")
                              ])

@app.callback(Output('output-prediction', 'children'),
              Input('upload-image', 'contents'))

def prediction(image):
    model = models.load_model('model.h5')
    def getCalssName(classNo):
        out = []
        for i in classNo:
            if i == 0:
                out.append("mask_weared_incorrect")
            elif i == 1:
                out.append("with_mask")
            elif i == 2:
                out.append("without_mask")
        return out
    curImg = cv2.imread(image)
    cur_Img = cv2.cvtColor(cur_Img, cv2.COLOR_RGB2BGR)
    curImg = np.array(curImg)
    curImg = cv2.resize(np.float32(curImg), (IMG_SIZE, IMG_SIZE))
    curImg = curImg.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    # %%
    predictions = model.predict(curImg)
    print(getCalssName(predictions[0]))
    return getCalssName(predictions[0])
#%%
def getCalssName(classNo):
  if   classNo == 0: return 'mask_worn_incorrect'
  elif classNo == 1: return 'with_mask'
  elif classNo == 2: return 'without_mask'
#%%
model = models.load_model("model.h5")
#%%
cur_Img = cv2.imread("/home/ubuntu/project/archive/Photo2.jpg")
cur_Img = cv2.cvtColor(cur_Img, cv2.COLOR_RGB2BGR)
plt.imshow(cur_Img)
plt.show()
cur_Img = np.array(cur_Img)
cur_Img = cv2.resize(np.float32(cur_Img), (IMG_SIZE, IMG_SIZE))
cur_Img = cur_Img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
predictions = model.predict(cur_Img)
print(getCalssName(predictions.argmax(axis=1)))