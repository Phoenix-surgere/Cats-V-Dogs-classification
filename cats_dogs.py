# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:19:20 2019

@author: black
"""

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Lambda
from keras.models import Sequential
import numpy as np

def train_cats_dogs_model():
    
    desired_accuracy = 0.95
    
    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') > desired_accuracy:
                print('Desired Accuracy (95%) reached, stop training')
                self.model.stop_training = True
                
    model = Sequential()
    model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x), 
                     input_shape=[90, 90,3]))
    model.add(Conv2D(32, (3,3),   activation='relu')) #input_shape=(100,100, 3),
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    callbacks = myCallback()
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    gennie = ImageDataGenerator(
            rescale= 1.0 / 255,
            rotation_range = 95,
            width_shift_range = 0.3,
            height_shift_range = 0.25,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
                            )
    train_it=gennie.flow_from_directory('cats_dogs/train/', class_mode='binary', 
                target_size=(90, 90), batch_size=64)
    val_it = gennie.flow_from_directory('cats_dogs/val/', class_mode='binary',
                                    target_size=(90, 90), batch_size=64)
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    
    history = model.fit_generator(train_it, epochs=10, callbacks=[callbacks],
                                  validation_data=val_it)
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and Validation Accuracy')
    plt.show()
    
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and Validation loss')
    plt.show()
    return history.history['acc'][-1]

train_cats_dogs_model()
