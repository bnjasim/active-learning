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
train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)

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

def define_model():
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # keep_prob1 = tf.placeholder(tf.float32)
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob1)
    t_size = 5*5*64
    # Dense layer
    W_fc1 = weight_variable([t_size, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2_drop, [-1, t_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # keep_prob2 = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)
    W_fc2 = weight_variable([128, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

y_conv1 = define_model()
cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv1, labels=y_))
train_step1 = tf.train.AdamOptimizer(2*1e-4).minimize(cross_entropy1)
correct_prediction1 = tf.equal(tf.argmax(y_conv1,1), tf.argmax(y_, 1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

y_conv2 = define_model()
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv2, labels=y_))
train_step2 = tf.train.AdamOptimizer(2*1e-4).minimize(cross_entropy2)
correct_prediction2 = tf.equal(tf.argmax(y_conv2,1), tf.argmax(y_, 1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

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

def train_tf_model(data, labels, step=None):
    print ('Training data size: ' + str(len(data)))
    start_time = time.time()
    batch_size = 100
    drop_train = 0.5
    for i in range(500):
        batch_x, batch_y = get_next_batch(data, labels, batch_size=batch_size)
        if i % 100 == 0:
            train_accuracy1 = sess.run(accuracy1, feed_dict={x:batch_x, y_: batch_y, keep_prob1:1.0, keep_prob2:1.0})
            print("step %d, training accuracy1 %g"%(i, train_accuracy1))
            
            train_accuracy2 = sess.run(accuracy2, feed_dict={x:batch_x, y_: batch_y, keep_prob1:1.0, keep_prob2:1.0})
            print("step %d, training accuracy2 %g"%(i, train_accuracy2))
        
        sess.run(train_step1, feed_dict={x: batch_x, y_: batch_y, keep_prob1:drop_train, keep_prob2:drop_train})
        sess.run(train_step2, feed_dict={x: batch_x, y_: batch_y, keep_prob1:drop_train, keep_prob2:drop_train})
     
    print ('Total time to train: ' + str(time.time() - start_time) + 's')
    
def test_tf_model(data, labels, step=None):
    print('Evaluate Model Test Accuracy after training')
    #summary, acc = sess.run([merged, accuracy], feed_dict={x:data, y_:labels, keep_prob1:1.0, keep_prob2:1.0})
    acc1 = sess.run(accuracy1, feed_dict={x:data, y_:labels, keep_prob1:1.0, keep_prob2:1.0})
    acc2 = sess.run(accuracy2, feed_dict={x:data, y_:labels, keep_prob1:1.0, keep_prob2:1.0})
    #if step:
    #    test_writer.add_summary(summary, step)
    #else:
    #    test_writer.add_summary(summary)
    # print('Test score:', score)
    print ('Test accuracy1:' + str(acc1))
    print ('Test accuracy2:' + str(acc2))
    return acc1

def clear_tf_model():
    sess.run(tf.global_variables_initializer())

def save_tf_model():
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./tmp/tf_model.ckpt")
    print("Model saved in path: %s" % save_path)

def restore_tf_model():
    saver = tf.train.Saver()
    # Restore variables from disk.
    saver.restore(sess, "./tmp/tf_model.ckpt")
    print("Model restored.")

n_samples = tf.placeholder(tf.int32)

out_prob1 = tf.nn.softmax(y_conv1)
max_prob1 = tf.reduce_max(out_prob1, axis=1)
topk1 = tf.nn.top_k(-max_prob1, n_samples)

out_prob2 = tf.nn.softmax(y_conv2)
max_prob2 = tf.reduce_max(out_prob2, axis=1)
topk2 = tf.nn.top_k(-max_prob2, n_samples)

def var_ratio_tf(pool_data, num_samples, step=None):
    # Var ratio active learning acquisition function
    t = sess.run(topk1, feed_dict={x: pool_data, keep_prob1:1.0, keep_prob2:1.0, n_samples:num_samples})
    return t.indices  

# design more interesting acquisition functions
# sort by sum of max_probs
max_prob = max_prob1 + max_prob2
topk = tf.nn.top_k(-max_prob, n_samples)

def var_ratio_sum(pool_data, num_samples, step=None):
    t = sess.run(topk, feed_dict={x: pool_data, keep_prob1:1.0, keep_prob2:1.0, n_samples:num_samples})
    return t.indices

# take disagreeing high probability predictions of both the models 
tf_labels1 = tf.argmax(y_conv1, axis=1)
tf_labels2 = tf.argmax(y_conv2, axis=1)
# mask =  tf.boolean_mask(max_prob, tf.equal(tf_labels1, tf_labels2))
# mask.assign(tf.zeros_like(mask))
# This is a trick to make the the same label prob values negative and keep the differing values as it was
max_diff = max_prob/2.0 - tf.cast(tf.equal(tf_labels1, tf_labels2), tf.float32)
topk_diff = tf.nn.top_k(max_diff, n_samples)

def var_ratio_diff(pool_data, num_samples, step=None):
    t = sess.run(topk_diff, feed_dict={x: pool_data, keep_prob1:1.0, keep_prob2:1.0, n_samples:num_samples})
    return t.indices

a = ActiveLearner(train_data, train_labels, test_data, test_labels, clear_tf_model, train_tf_model, test_tf_model, save_tf_model, restore_tf_model, init_num_samples=20)
out_acc = a.run(20, [var_ratio_diff, var_ratio_tf], pool_subset_count=1000)
# a.experiment(50, [var_ratio_diff, var_ratio_tf], pool_subset_count=1000)
a.plot()