from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GaussianNoise
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import pandas as pd
import random
import os
import shutil
import matplotlib.pyplot as plt


#Load the model for ensemble prediction
model_Xcep = load_model("drive/My Drive/Dataset/Xception_model_weights.h5")

model_IncV3 = load_model("drive/My Drive/Dataset/InceptionV3_model_weights.h5")

model_IncResE150 = load_model("drive/My Drive/Dataset/InceptionResnetV2_model_weights.h5")

models = [model_Xcep, model_IncV3, model_IncResE150]

def ensemble_predictions(models_members, image_path):
  #print (image_path)
  img = image.load_img(image_path, target_size=(WIDTH, HEIGHT))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  images = np.vstack([x])
  
	# make predictions
  ensemble_pred = [model.predict(images) for model in models_members]
  mean = np.mean(ensemble_pred, axis=0)
  result = np.argmax(mean, axis=1)
  print (class_map.get(result[0]))
  return class_map.get(result[0])


# Ensemble Prediction
mypath = os.getcwd() + TEST_DIR
test_filenames = os.listdir(mypath)
print (len(test_filenames))
print(test_filenames)


#Prediction
result_dict = {}
for filename in test_filenames:
  image_path = "drive/My Drive/Dataset/Food_images/test/" + filename
  #print (image_path)
  #predict_model(image_path)
  print(image_path)
  result_dict[filename] = ensemble_predictions(models, image_path)



"""**Exporting the result for the submission**"""

df = pd.DataFrame.from_dict(result_dict, orient='index')
df = df.reset_index()
df.columns =['file' , 'label']
df.sort_values(by=['file'], inplace=True)

result_filename = MODEL_NAME + '.tsv'
df.to_csv(result_filename, sep='\t' ,index=False)
