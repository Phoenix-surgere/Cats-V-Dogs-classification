# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:17:55 2019

@author: black
"""
import matplotlib.pyplot as plt
#import numpy as np
phoenix = plt.imread('fantasy_phoenix-wallpaper-1920x1080.jpg')
plt.imshow(phoenix)
plt.show()

import tensorflow as tf
print(tf.__version__)
#import skimage
#mnist = tf.keras.datasets.mnist

#(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

#training_images = training_images/255.0
#test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#model.compile(optimizer = 'adam',
#              loss = 'sparse_categorical_crossentropy')
#
#model.fit(training_images, training_labels, epochs=1)
#
#model.evaluate(test_images, test_labels)
#
#classifications = model.predict(test_images)

from skimage import data, color; rocket=data.rocket()#; plt.imshow(rocket)

def show_image(image, title='Title', cmap_type='gray'):
    plt.imshow(image,cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation='nearest', cmap='gray_r')
    plt.title('Contours')
    plt.axis('off')
    plt.show()

def show_image_with_corners(image, coords, title="Corners detected"):
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
    plt.axis('off')
    plt.show()


#How to inspect function source code    
import inspect
lines = inspect.getsource(show_image_with_corners)
print(lines)
    
#show_image(rocket)
#show_image(color.rgb2gray(rocket))
red = phoenix[:,:,0]; green = phoenix[:,:,1]; blue=phoenix[:,:,2]
plt.hist(red.ravel(), bins=256);
plt.ylabel('Pixel Frequencies'); plt.xlabel('Pixel Intensities') 
plt.title('Red Pixels'); plt.show()
plt.hist(green.ravel(), bins=256); 
plt.ylabel('Pixel Frequencies'); plt.xlabel('Pixel Intensities') 
plt.title('Green Pixels'); plt.show()
plt.hist(blue.ravel(), bins=256) 
plt.ylabel('Pixel Frequencies'); plt.xlabel('Pixel Intensities') 
plt.title('Blue Pixels'); plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator
