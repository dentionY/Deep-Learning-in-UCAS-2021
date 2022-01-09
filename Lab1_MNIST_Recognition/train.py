# Author Yang Dengtian
# Time   2021/05/10
# Version 1 ---- 随机梯度下降太慢
# Version 2 ---- Adam优化器，增加drop
# Version 3 ---- 若干参数调整
# Reference ： https://blog.csdn.net/sinat_34328764/article/details/83832487 主要学习如何导入mnist并处理成独热码
# Reference ： https://blog.csdn.net/limiyudianzi/article/details/84960074   主要学习优化器的类型和原理，以及参考了知乎上关于选参数的经验
# Reference ： 其余包括模型保存和加载的学习
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
import numpy as np

image = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(image,[-1,28,28,1])

#第一层卷积层参数
conv1_weight = tf.Variable(tf.truncated_normal([5, 5, 1, 32],stddev = 0.1))
conv1_bias = tf.Variable(tf.constant(0.1,shape = [32]))
#第二层卷积层参数
conv2_weight = tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev = 0.1))
conv2_bias = tf.Variable(tf.constant(0.1,shape = [64]))
#第一层全连接层参数
fc1_weight = tf.Variable(tf.truncated_normal([7*7*64, 1024],stddev = 0.1))
fc1_bias = tf.Variable(tf.constant(0.1,shape = [1024]))
#第二层全连接层参数
fc2_weight = tf.Variable(tf.truncated_normal([1024, 10],stddev = 0.1))
fc2_bias = tf.Variable(tf.constant(0.1,shape = [10]))
#第一层卷积层
conv1 = tf.nn.conv2d(x_image, conv1_weight, strides = [1,1,1,1], padding = 'SAME')
conv1 = tf.nn.relu(conv1 + conv1_bias)
pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#第二层卷积层
conv2 = tf.nn.conv2d(pool1, conv2_weight, strides = [1,1,1,1], padding = 'SAME')
conv2 = tf.nn.relu(conv2 + conv2_bias)
pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#展平
pool2 = tf.reshape(pool2, [-1, 7*7*64])
#第一层全连接层
fc1 = tf.nn.relu(tf.matmul(pool2, fc1_weight) + fc1_bias)
fc1 = tf.nn.dropout(fc1, keep_prob)
#输出层
label_pred=tf.nn.softmax(tf.matmul(fc1, fc2_weight) + fc2_bias)

cross_entropy = -tf.reduce_sum(label*tf.log(label_pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = tf.equal(tf.argmax(label_pred,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

saver = tf.train.Saver()
train_acc_box = []
train_ind_box = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(15000):
        batch = mnist.train.next_batch(100)
        if i % 300 == 0:
            train_accuracy = accuracy.eval(feed_dict={image: batch[0], label: batch[1], keep_prob: 0.5})
            print('Step %d, training accuracy %g' % (i, train_accuracy))
            train_acc_box.append(train_accuracy)
            train_ind_box.append(i)
        train_step.run(feed_dict={image: batch[0], label: batch[1], keep_prob: 0.5})
    saver.save(sess, 'D:/AI computer system Lab/MyLab/DLcourse/Lab1/NewLab1_1/SAVE/model.ckpt')

from matplotlib import pyplot as plt 

plt.title("Training acc--Training iter") 
plt.xlabel("Training iter") 
plt.ylabel("Training acc") 
plt.plot(train_ind_box,train_acc_box) 
plt.show()