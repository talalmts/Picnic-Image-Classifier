# -*- coding: utf-8 -*-
"""Image Classification with Pre-trained Models - Picnic

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XMyvkusJIpWp0MLIqagNNNUywI_oPDyz
"""

#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input, decode_predictions

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

"""**Loading the dataset**"""

#mount the drive
from google.colab import drive
drive.mount('/content/drive')

"""**Data Augmentation**"""

TRAIN_DIR = "drive/My Drive/Dataset/Food_images/Train"
VALIDATION_DIR = "drive/My Drive/Dataset/Food_images/Validation/"
TEST_DIR = "/drive/My Drive/Dataset/Food_images/test/"
HEIGHT = 299
WIDTH = 299
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)



train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                   target_size=(HEIGHT, WIDTH),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical',
                                                   shuffle=True,
                                                   color_mode='rgb')

#Need to construct a validation set for the set
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)


validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    color_mode='rgb'
)

"""**Display the Augmented images**"""


#name of the target classes
class_list = ['Bananas, apples & pears', 'Berries & cherries',
       'Pork, beef & lamb', 'Bell peppers, zucchinis & eggplants',
       'Pudding, yogurt & quark', 'Minced meat & meatballs', 'Eggs',
       'Cucumber, tomatoes & avocados', 'Citrus fruits', 'Fish',
       'Onions, leek, garlic & beets', 'Kiwis, grapes & mango',
       'Fresh bread', 'Cheese', 'Salad & cress', 'Pre-baked breads',
       'Fresh herbs', 'Nectarines, peaches & apricots',
       'Lunch & Deli Meats', 'Milk', 'Potatoes',
       'Broccoli, cauliflowers, carrots & radish',
       'Asparagus, string beans & brussels sprouts', 'Poultry',
       'Pineapples, melons & passion fruit']

## create mapping from for prediction score
classes = train_generator.class_indices 
class_map = {value:key for key,value in classes.items()}
#print (class_map)

"""**Loading the pre-trained model**"""

# load the pre-trained model
base_model = Xception(weights='imagenet', 
                         include_top=False, 
                         input_shape=(HEIGHT,WIDTH,3))

#Fine-tuning the pretrained model using the dataset
def build_finetune_model(base_model, dropout,fc_layers, num_classes, fine_tune_at):
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    x = base_model.output
    #x = Flatten()(x)
    x = GlobalAveragePooling2D(name='max_pool')(x)
    x = GaussianNoise(0.1)(x)
    
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc,activation='relu')(x)
        x = Dropout(dropout)(x)
    
    x = Dense(64, input_dim=64,kernel_regularizer=regularizers.l2(0.01))(x)
    # New Softmax layer
    prediction = Dense(num_classes, activation='softmax')(x)
    
    finetune_model = Model(input=base_model.input, output=prediction)
    
    return finetune_model

FC_LAYERS = [1024,1024]
dropout = 0.5
print (len(base_model.layers))

finetune_model = build_finetune_model(base_model,
                                     dropout=dropout,
                                     fc_layers=FC_LAYERS,
                                     num_classes=len(class_list),
                                     fine_tune_at=60)

#print (finetune_model.summary())

"""**Training the model**"""

# hyperparameter of the model
NUM_EPOCHS = 100
BATCH_SIZE = 32
num_train_images = len(train_generator.index_array)
num_validation_images = len(validation_generator.index_array)

MODEL_NAME = "Xception"

MODEL_FILE = 'drive/My Drive/Dataset/' + MODEL_NAME + '.model'

#optimizier for the network
adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', 
                       metrics=['accuracy'])

#print (finetune_model.summary())

filepath="drive/My Drive/Dataset/" + MODEL_NAME + "_model_weights.h5"

checkpoint = [EarlyStopping(monitor='val_acc', patience=5, verbose=2),ModelCheckpoint(filepath, monitor="val_acc", verbose=1, mode='auto', save_best_only=True)]
callbacks_list = checkpoint

history = finetune_model.fit_generator(train_generator, 
                                       epochs=NUM_EPOCHS, 
                                       workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
                                       shuffle=True, 
                                       callbacks=callbacks_list,
                                       validation_data=validation_generator,
                                       validation_steps=num_validation_images // BATCH_SIZE)

finetune_model.save(filepath)

"""**Model Evaluation**"""

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
#plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""****Prediction using the above trained model****"""


def predict_model(image_path):
  img = image.load_img(image_path, target_size=(WIDTH, HEIGHT))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  images = np.vstack([x])
  y_prob = finetune_model.predict(images)
  y_classes = y_prob.argmax(axis=-1)
  print (class_map.get(y_classes[0]))
  return class_map.get(y_classes[0])

mypath = os.getcwd() + TEST_DIR
test_filenames = os.listdir(mypath)
print (len(test_filenames))
print(test_filenames)


#Prediction
result_dict = {}
for filename in test_filenames:
  image_path = "drive/My Drive/Dataset/Food_images/test/" + filename
  print(image_path)
  result_dict[filename] = predict_model(image_path)


"""**Exporting the result for the submission**"""

df = pd.DataFrame.from_dict(result_dict, orient='index')
df = df.reset_index()
df.columns =['file' , 'label']
df.sort_values(by=['file'], inplace=True)

result_filename = MODEL_NAME + '.tsv'
df.to_csv(result_filename, sep='\t' ,index=False)
