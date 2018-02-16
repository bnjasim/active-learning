import numpy as np
import tensorflow as tf
import time

# import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/Users/bnjasim/Desktop/active learning/Experiments/MNIST_data/", one_hot=True)

import imp
imp.load_source('activelearn', '../activelearn.py')
from activelearn import *

train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

keep_prob1 = tf.placeholder(tf.float32)
h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob1)

# Dense layer
t_size = 5*5*64
W_fc1 = weight_variable([t_size, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2_drop, [-1, t_size])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob2 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)

W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def get_next_batch(data, labels, batch_size):
    if len(data) != len(labels):
        raise Exception('data and labels should have same number of elements')
    
    if batch_size > len(data):
        batch_size = len(data)
        #raise Exception('batch size is more than size of data')
    
    indices = np.random.choice(range(len(data)), batch_size, replace=False)
    return data[indices], labels[indices]

def train_tf_model(data, labels):
    print ('Training data size: ' + str(len(data)))
    start_time = time.time()
    batch_size = 100
    drop_train = 0.5
    for i in range(50): # 500 works best
        batch_x, batch_y = get_next_batch(data, labels, batch_size=batch_size)
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_: batch_y, keep_prob1:1.0, keep_prob2:1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob1:drop_train, keep_prob2:drop_train})
     
    # print("test accuracy %g"%sess.run(accuracy, feed_dict={x: test_data, y_: test_labels, keep_prob1:1.0, keep_prob2:1.0}))
    print ('Total time to train: ' + str(time.time() - start_time) + 's')

def test_tf_model(data, labels):
    print('Evaluate Model Test Accuracy after training')
    acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels, keep_prob1:1.0, keep_prob2:1.0})
    # print('Test score:', score)
    print ('Test accuracy:' + str(acc))
    return acc

def clear_tf_model():
    sess.run(init)

def save_tf_model():
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./tmp/tf_model.ckpt")
    print("Model saved in path: %s" % save_path)

def restore_tf_model():
    saver = tf.train.Saver()
    # Restore variables from disk.
    saver.restore(sess, "./tmp/tf_model.ckpt")
    print("Model restored.")

out_prob = tf.nn.softmax(y_conv)

def var_ratio_tf(pool_data):
    # Var ratio active learning acquisition function
    D_probs = sess.run(out_prob, feed_dict={x: pool_data, keep_prob1:1.0, keep_prob2:1.0})
    return 1.0 - np.max(D_probs, axis=1)


a = ActiveLearner(train_data, train_labels, test_data, test_labels, clear_tf_model, train_tf_model, test_tf_model, save_tf_model, restore_tf_model, init_num_samples=20)
# a.run(2, [random_acq, var_ratio_tf], pool_subset_count=1000)
var_acc = a.experiment(50, [random_acq, var_ratio_tf], pool_subset_count=1000, num_exp=3)
# var_acc = a.experiment(2, [random_acq, var_ratio_tf], pool_subset_count=1000, num_exp=1)

a.plot()
