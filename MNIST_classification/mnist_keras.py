import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/Users/bnjasim/Desktop/active learning/Experiments/MNIST_data/", one_hot=True)

import imp
imp.load_source('activelearn', '../activelearn.py')
from activelearn import *


batch_size     = 32
nb_classes     = 10
nb_epochs     = 10
img_rows, img_cols = 28,28
nb_filters     = 32
pool_size      = 2
kernel_size    = 3
input_shape    = (img_rows, img_cols, 1)

train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

# Reshape to 28x28
train_data = train_data.reshape(len(train_data),  img_rows, img_cols, 1)
test_data = test_data.reshape(len(test_data),  img_rows, img_cols, 1)

def define_model():
    global  model
    model = Sequential()
    model.add(Convolution2D(filters= nb_filters, kernel_size=(kernel_size,kernel_size), 
                            input_shape = input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

    model.add(Convolution2D(filters=nb_filters, kernel_size=(kernel_size,kernel_size),activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))

    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128 , activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


def train_keras(data, labels):
    print ('Training data size: ' + str(len(data)))
    model.fit(data, labels, batch_size=batch_size, epochs=nb_epochs, verbose=1)

def test_keras(data, labels):
    print('Evaluate Model Test Accuracy after training')
    score, acc = model.evaluate(data, labels, verbose=1)
    # print('Test score:', score)
    print ('\nTest accuracy:' + str(acc))
    return acc


a = ActiveLearner(train_data, train_labels, test_data, test_labels, define_model, train_keras, test_keras)

# a.run(20, var_ratio, pool_subset_count=1000)
a.run(10, random_acq, pool_subset_count=1000)
a.plot()
