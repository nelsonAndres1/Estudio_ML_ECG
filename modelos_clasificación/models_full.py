# -*- coding: utf-8 -*-
"""
Created on Thu May 28 00:39:10 2020

@author: JARS
"""

import tensorflow as tf  
import keras.backend as K
from keras import layers
from keras import Model
from tensorflow.keras.optimizers import SGD

def train_validation_data(data_directory, model, height, width, batch_size, class_type, val_split):
    #from keras.preprocessing.image import ImageDataGenerator
    ImageDataGenerator=tf.keras.preprocessing.image.ImageDataGenerator
    if model=='in_resV2':
        #from keras.applications.inception_resnet_v2 import preprocess_input
        pre_function=tf.keras.applications.inception_resnet_v2.preprocess_input
    elif model=='Xception':
        #from keras.applications.xception import preprocess_input
        pre_function=tf.keras.applications.xception.preprocess_input
    else:
        pre_function=None

    train_datagen = ImageDataGenerator(  
    rescale=1. / 255,
    #width_shift_range = 0.05,
    #height_shift_range = 0.05,
    #horizontal_flip = True,
    #vertical_flip = True,
    #rotation_range = 20, 
    #zoom_range = 0.05,
    #shear_range = 0.05,
    #fill_mode = "nearest",
    validation_split=val_split,
    preprocessing_function=pre_function)
    
    val_datagen=ImageDataGenerator(rescale=1. / 255,     
                               validation_split=val_split,
                               preprocessing_function=pre_function)
    
    train_generator = train_datagen.flow_from_directory(  
    data_directory,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode=class_type,
    subset='training',
    shuffle=True)
     
    validation_generator = val_datagen.flow_from_directory(  
    data_directory,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode=class_type,
    subset='validation',
    shuffle=False)
    
    return train_generator, validation_generator

def unfreeze(model, a, b, stage):
    if stage=='val':
        i=0
        for layer in model.layers:
            if i>a*len(model.layers)//b:
                layer.trainable=True
                if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance') and (model=='Xception' or model=='in_resV2'):
                  K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
                  K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
            else:
                layer.trainable=False
            i=i+1
    else:
        for layer in model.layers:
          if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance') and model=='Xception' or model=='in_resV2':
              layer.trainable = True
              K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
              K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
          else:
              layer.trainable = False

          
    
def model_TL(model,  height, width, channels): #models= VGG, InceptionResNETV2, MobileNet, Xception, Dense
    K.set_learning_phase(0)
    if model=='in_resV2':
        #from keras.applications import InceptionResNetV2
        pre_trained_model = tf.keras.applications.InceptionResNetV2(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='incep':
        #from keras.applications import MobileNetV2
        pre_trained_model = tf.keras.applications.InceptionV3(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='Xception':
        #from keras.applications import xception    
        pre_trained_model = tf.keras.applications.Xception(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='VGG':
        #from keras.applications import VGG16
        pre_trained_model = tf.keras.applications.VGG16(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='dense':
        #from keras.applications import DenseNet201
        pre_trained_model = tf.keras.applications.DenseNet201(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
    elif model=='resnet':
        #from efficient.keras import efficient
        pre_trained_model = tf.keras.applications.ResNet101V2(
            weights = "imagenet",#NULL,#imagenet", Random weight
            include_top = False,
            input_shape = (height, width, channels)
          )
   
    #pre_trained_model.summary()
    if model=='cnn':
      pass
    else:
      unfreeze(pre_trained_model, 3,4, stage='train')
    # i=0
    # for layer in pre_trained_model.layers:
    #     if i<3*len(pre_trained_model.layers)//4:
    #         layer.trainable=False
    #     else:
    #         layer.trainable=True
    #     i=i+1


# Configure and compile the model
    K.set_learning_phase(1)
    if model=='cnn':
        #final_model=tf.keras.models.Sequential([
        #tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
        #                        input_shape = (height, width, channels)),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
        #                        input_shape = (height, width, channels)),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu',
        #                        input_shape = (height, width, channels)),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(512,activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(1,activation='sigmoid'),
        #])
        img_input = tf.keras.layers.Input(shape=(height, width, channels))
        model = tf.keras.layers.BatchNormalization(axis = 3)(img_input)
        model = tf.keras.layers.Convolution2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model)
        model = tf.keras.layers.MaxPooling2D()(model)
    
        model = tf.keras.layers.BatchNormalization(axis = 3)(model)
        model = tf.keras.layers.Convolution2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model)
        model = tf.keras.layers.MaxPooling2D()(model)
    
        model = tf.keras.layers.BatchNormalization(axis = 3)(model)
        model = tf.keras.layers.Convolution2D(filters = 128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model)
        model = tf.keras.layers.MaxPooling2D()(model)
    
        model = tf.keras.layers.BatchNormalization(axis = 3)(model)
        model = tf.keras.layers.Convolution2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model) 
        model = tf.keras.layers.GlobalAveragePooling2D()(model)
        #x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(img_input)
        #x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)

        #x = tf.keras.layers.MaxPooling2D(2)(x)

        #x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        #x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)

        #x = tf.keras.layers.MaxPooling2D(2)(x)

        #x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        #x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)

        #x = tf.keras.layers.MaxPooling2D(2)(x)

        #x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        #x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)


        #x = tf.keras.layers.MaxPooling2D(2)(x)


        #x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        #x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)

        #x = tf.keras.layers.MaxPooling2D(2)(x)

     
      
        #x = tf.keras.layers.Dropout(0.5)(x)


        #x = tf.keras.layers.Flatten()(x) 
        
        #x = tf.keras.layers.Dense(128, activation='relu')(x)
        #x = tf.keras.layers.Dropout(0.5)(x)
        #model=tf.keras.layers.Dense(64, activation='elu')(model)

        model = tf.keras.layers.Dropout(0.5)(model)
        #x = tf.keras.layers.Dropout(0.5)(x)


        output = tf.keras.layers.Dense(1, activation='sigmoid')(model)
        final_model = tf.keras.models.Model(img_input, output)

    else:
        final_model=tf.keras.models.Sequential([
        pre_trained_model,
        #tf.keras.layers.GlobalMaxPooling2D(),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(128, activation='elu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(128, activation='elu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(128, activation='elu'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid'),
        ])


    final_model.summary()
    return final_model

def training_model(model, epochs, batch_size, learning_rate, training_data, validating_data, patience, stage, model_name):


    es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=patience)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model_'+model_name+'.h5', 
                         monitor='accuracy', mode='max', verbose=1, save_best_only=True)
    callbakcs_mdl=[es, mc]
    
    if stage=='val':
        #model.trainable=True
        learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)
        callbakcs_mdl.append(learning_rate_reduction)
        #callbakcs_mdl[2]=learning_rate_reduction
        unfreeze(model, 3,4, stage='val')
        
    model.compile(
    #optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate),#(lr = 0.001,  beta_1=0.9, beta_2=0.999),

    optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),#Adam(lr = learning_rate,  beta_1=0.9, beta_2=0.999),
    #optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99),
    #loss='categorical_crossentropy',
    loss='binary_crossentropy',
    metrics= ["accuracy"])
    
   
    history=model.fit_generator(
    training_data,
    steps_per_epoch = training_data.n//batch_size+1,  
    epochs = epochs,
    validation_data = validating_data,
    validation_steps = validating_data.n//batch_size+1,
    callbacks=callbakcs_mdl)
    
    return history

