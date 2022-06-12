
#%% IMPORTING LIBRARIES 
import os
import glob
import shutil
import json
import keras
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Defining the working directories

work_dir = r'C:\Users\folio 1040'
os.listdir(work_dir) 
train_path = r'C:\Users\folio 1040\train_images'

#%% IMPORTING DATA

# Importing train.csv

data = pd.read_csv(work_dir + r'\train.csv')
print(Counter(data['label']))

data['label'].hist()

# Importing the json file with labels

f = open(work_dir + r'\label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k):v for k,v in real_labels.items()}

# Defining the working dataset
data['class_name'] = data.label.map(real_labels)
print(data.head(10))
print(data['class_name'].unique())



def showImages(images):

    # Extract 16 random images from it
    random_images = [np.random.choice(images) for i in range(16)]

    # Adjust the size of your images
    plt.figure(figsize=(16,12))

    # Iterate and plot random images
    for i in range(16):
        plt.subplot(4,4, i + 1)
        img = plt.imread(train_path+'/'+random_images[i])
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()  


mask = data['label'] ==4
classHealthy = data[mask]

showImages(classHealthy['image_id'])


mask = data['label'] ==3
classCMD = data[mask]

showImages(classCMD['image_id'])


mask = data['label'] ==2
classCGM = data[mask]

showImages(classCGM['image_id'])

mask = data['label'] ==1
classCBSD = data[mask]

showImages(classCBSD['image_id'])


mask = data['label'] ==0
classCBB = data[mask]

showImages(classCBB['image_id'])


class0 = classCBB.sample(frac=0.99)
class1 = classCBSD.sample(frac=0.9)
class2 = classCGM.sample(frac=0.9)
class3 = classCMD.sample(frac=0.9)
class4 = classHealthy.sample(frac=0.9)




frames=[class0,class1,class2,class3,class4]
finalData = pd.concat(frames)
finalData.head(10)
print(len(finalData))

# Spliting the data
from sklearn.model_selection import train_test_split

train,val = train_test_split(finalData, test_size = 0.05, random_state = 42, stratify = finalData['class_name'])

# Importing the data using ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
size = (IMG_SIZE,IMG_SIZE)
n_CLASS = 5

datagen = ImageDataGenerator(
                    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
                    rotation_range = 60,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = True,
                    fill_mode = 'nearest')

train_set = datagen.flow_from_dataframe(train,
                         directory = train_path,
                        #  seed=42,
                         x_col = 'image_id',
                         y_col = 'class_name',
                         target_size = size,
                         class_mode = 'categorical',
                         interpolation = 'nearest',
                         shuffle = True,
                         batch_size = 32)

val_set = datagen.flow_from_dataframe(val,
                         directory = train_path,
                        #  seed=42,
                         x_col = 'image_id',
                         y_col = 'class_name',
                         target_size = size,
                         class_mode = 'categorical',
                         interpolation = 'nearest',
                         shuffle = True,
                         batch_size = 32)


def create_model():
    
    model = Sequential()
    # initialize the model with input shape as (224,224,3)
    model.add(tf.keras.applications.EfficientNetB3(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top = False, weights = 'imagenet' ))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dropout(0.7))
    model.add(Dense(32, activation = 'relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dropout(0.7))
    model.add(Dense(n_CLASS, activation = 'softmax'))
    
    return model

leaf_model = create_model()
leaf_model.summary()

EPOCHS = 5
STEP_SIZE_TRAIN = train_set.n//train_set.batch_size
STEP_SIZE_VALID = val_set.n//val_set.batch_size


#%% FITTING THE MODEL

def Model_fit():
    
    #leaf_model = None
    
    leaf_model = create_model()
    
    '''Compiling the model'''
    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False,
                                                   label_smoothing=0.001,
                                                   name='categorical_crossentropy' )
    
    leaf_model.compile(optimizer = Adam(learning_rate = 2e-4),
                        loss = loss, #'categorical_crossentropy'
                        metrics = ['categorical_accuracy']) #'acc'
    
    # Stop training when the val_loss has stopped decreasing for 5 epochs.
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                       restore_best_weights=True, verbose=1)
    
    # Save the model with the minimum validation loss
    checkpoint_cb = ModelCheckpoint("Cassava_best_modelEffNetB3.h5",
                                    save_best_only=True,
                                    monitor = 'val_loss',
                                    mode='min')
    
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.3,
                                  patience = 3,
                                  min_lr = 1e-6,
                                  mode = 'min',
                                  verbose = 1)
    
    history = leaf_model.fit(train_set,
                             validation_data = val_set,
                             epochs= EPOCHS,
                             batch_size = 32,
                             steps_per_epoch = STEP_SIZE_TRAIN,
                             validation_steps = STEP_SIZE_VALID,
                             callbacks=[es, checkpoint_cb, reduce_lr])
    
    leaf_model.save('Cassava_modelEffNetB3'+'.h5')  
    
    return history


history = Model_fit()


acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


import keras

final_model = keras.models.load_model('Cassava_best_modelEffNetB3.h5')


# Importing the json file with labels

f = open(work_dir + r'\label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k):v for k,v in real_labels.items()}

#  Defining the working dataset
data['class_name'] = data.label.map(real_labels)

class0 = classCBB.sample(frac=0.99)
class1 = classCBSD.sample(frac=0.7)
class2 = classCGM.sample(frac=0.7)
class3 = classCMD.sample(frac=0.7)
class4 = classHealthy.sample(frac=0.7)




frames=[class0,class1,class2,class3,class4]
finalData = pd.concat(frames)
finalData.head(10)
print(len(finalData))

# Spliting the data
from sklearn.model_selection import train_test_split

train,val = train_test_split(finalData, test_size = 0.3, random_state = 40, stratify = finalData['class_name'])

# Importing the data using ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128 
size = (IMG_SIZE,IMG_SIZE)
n_CLASS = 5

datagen = ImageDataGenerator(
                    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
                    rotation_range =40,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = True,
                    fill_mode = 'nearest')

train_set = datagen.flow_from_dataframe(train,
                         directory = train_path,
                        #  seed= 40,
                         x_col = 'image_id',
                         y_col = 'class_name',
                         target_size = size,
                         #color_mode="rgb",
                         class_mode = 'categorical',
                         interpolation = 'nearest',
                         shuffle = True,
                         batch_size = 64)
valDatagen = ImageDataGenerator(
                    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input
                    )

val_set = valDatagen.flow_from_dataframe(val,
                         directory = train_path,
                        #  seed=40,
                         x_col = 'image_id',
                         y_col = 'class_name',
                         target_size = size,
                         #color_mode="rgb",
                         class_mode = 'categorical',
                         interpolation = 'nearest',
                         shuffle = True,
                         batch_size = 64)



EPOCHS = 3
STEP_SIZE_TRAIN = train_set.n//train_set.batch_size
STEP_SIZE_VALID = val_set.n//val_set.batch_size


def Model_fit1(model):
    
    leaf_model = model
    
    
    '''Compiling the model'''
    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False,
                                                   label_smoothing=0.01,
                                                   name='categorical_crossentropy' )
    
    leaf_model.compile(optimizer =  Adam(learning_rate = 2e-4),
                        loss = loss, #'categorical_crossentropy'
                        metrics = ['categorical_accuracy']) #'acc'
    
    # Stop training when the val_loss has stopped decreasing for 5 epochs.
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                       restore_best_weights=True, verbose=1)
    
    # Save the model with the minimum validation loss
    checkpoint_cb = ModelCheckpoint("Cassava_best_modelEffNetB3v3.h5",
                                    save_best_only=True,
                                    monitor = 'categorical_accuracy',
                                    mode='max')
    
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.1,
                                  patience = 2,
                                  min_lr = 1e-6,
                                  mode = 'min',
                                  verbose = 1)
    
    history = leaf_model.fit(train_set,
                             validation_data = val_set,
                             epochs= EPOCHS,
                             batch_size = 32,
                             steps_per_epoch = STEP_SIZE_TRAIN,
                             validation_steps = STEP_SIZE_VALID,
                             callbacks=[es, checkpoint_cb, reduce_lr])
    
    leaf_model.save('Cassava_modelEffNetB3v3'+'.h5')  
    
    return history


Model_fit1(final_model) 


