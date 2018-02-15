import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time

import imp
imp.load_source('activelearn', '../activelearn.py')
from activelearn import *

# import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/",one_hot=True)

# unroll rnn
time_steps = 28
# hidden LSTM units
num_units = 128
# rows of 28 pixels
n_input = 28
# learning rate for adam
learning_rate = 0.001
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 10
# size of batch
batch_size=32

train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

# Reshape to 28x28
train_data = train_data.reshape(len(train_data), 28, 28)
test_data = test_data.reshape(len(test_data), 28, 28)

# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# defining placeholders
# input image placeholder
x = tf.placeholder("float",[None, time_steps, n_input])
# input label placeholder
y = tf.placeholder("float",[None, n_classes])

# processing the input tensor from [batch_size,n_steps,n_input]
# to "time_steps" number of [batch_size,n_input] tensors
input = tf.unstack(x, time_steps, 1)

# defining the network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs,_ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()

def get_next_batch(data, labels, batch_size):
    if len(data) != len(labels):
        raise Exception('data and labels should have same number of elements')
    
    if batch_size > len(data):
        batch_size = len(data)
        #raise Exception('batch size is more than size of data')
    
    indices = np.random.choice(range(len(data)), batch_size, replace=False)
    return data[indices], labels[indices]


def train_tf_lstm(data, labels):
    print ('Training data size: ' + str(len(data)))
    start_time = time.time()
    batch_size = 100
    drop_train = 0.5
    for i in range(500):
        batch_x, batch_y = get_next_batch(data, labels, batch_size=batch_size)
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y: batch_y})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
     
    print("test accuracy %g"%sess.run(accuracy, feed_dict={x: test_data, y: test_labels}))
    print ('Total time to train: ' + str(time.time() - start_time) + 's')


def test_tf_lstm(data, labels):
    print('Evaluate Model Test Accuracy after training')
    acc = sess.run(accuracy, feed_dict={x: test_data, y: test_labels})
    # print('Test score:', score)
    print ('Test accuracy:' + str(acc))
    return acc

def clear_tf_model():
    sess.run(init)

def save_tf_model():
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./tmp/tf_lstm_model.ckpt")
    print("Model saved in path: %s" % save_path)

# save_tf_model()

# clear_tf_model()
# test_tf_model(test_data, test_labels)

# restore_tf_model()

def restore_tf_model():
    saver = tf.train.Saver()
    # Restore variables from disk.
    saver.restore(sess, "./tmp/tf_lstm_model.ckpt")
    print("Model restored.")

out_prob = tf.nn.softmax(prediction)

def var_ratio_tf(pool_data):
    # Var ratio active learning acquisition function
    D_probs = sess.run(out_prob, feed_dict={x: pool_data})
    return 1.0 - np.max(D_probs, axis=1)

def random_acq(pool_data):
    return np.random.rand(len(pool_data)) 

a = ActiveLearner(train_data, train_labels, test_data, test_labels, clear_tf_model, train_tf_lstm, test_tf_lstm, save_tf_model, restore_tf_model, init_num_samples=100)
var_acc = a.experiment(100, [random_acq, var_ratio_tf], pool_subset_count=1000)