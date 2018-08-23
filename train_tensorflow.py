#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:51:57 2018

@author: picot
"""


import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

###############################################################################
#                                                                             #
#   Basic Classification                                                      #
#                                                                             #
#   https://www.tensorflow.org/tutorials/keras/basic_classification           #
#                                                                             #
###############################################################################


fashion_mnist = keras.datasets.fashion_mnist

(train_images_basic, train_labels_basic), (test_images_basic, test_labels_basic) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images_basic = train_images_basic / 255.0
test_images_basic = test_images_basic / 255.0



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images_basic, train_labels_basic, epochs=5)
predictions = model.predict(test_images_basic)


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images_basic[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels_basic[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
    
    
###############################################################################
#                                                                             #
#   Text Classification                                                       #
#                                                                             #
#   https://www.tensorflow.org/tutorials/keras/basic_text_classification      #
#                                                                             #
###############################################################################

    
imdb = keras.datasets.imdb

(train_data_text, train_labels_text), (test_data_text, test_labels_text) = imdb.load_data(num_words=10000)

    
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

    
    
    
    
    
    
    
    
    
    
    
    
    