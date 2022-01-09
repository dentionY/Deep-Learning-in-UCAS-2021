# 0.718 10000张图片  10000steps  3层卷积层 未修改参数
# 0.718 15000张图片  15000steps  3层卷积层 未修改参数
# 0.712 10000张图片  15000steps  3层卷积层 未修改参数
# 0.702 10000张图片  15000steps  3层卷积层  修改参数1024
# 0.684 10000张图片  15000steps  3层卷积层 去除尾层的池化&标准化 未修改参数
# 0.690 10000张图片  15000steps  4层卷积层 
# 0.78  final

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
import os 
import trainCD

test_name = "D:\\AI computer system Lab\\MyLab\\DLcourse\\Lab2\\New\\test_sample.txt"
test_label_name = "D:\\AI computer system Lab\\MyLab\\DLcourse\\Lab2\\New\\test_label.txt"
test_box = []
with open(test_name, 'r') as file_to_read:
    all_file = file_to_read.readlines()
    for row in all_file:
        row_list = list(row)
        row_list[-1] = row_list[-1].replace('\n','')
        row_result = ''.join(row_list)
        test_box.append(row_result)

cat_or_dog_box = []
with open(test_label_name, 'r') as testfile_to_read:
    test_all_file = testfile_to_read.readlines()
    for row in test_all_file:
        if row == 'cat\n':
            cat_or_dog_box.append(1.)
        elif row == 'dog\n':
            cat_or_dog_box.append(0.)
 
#print(cat_or_dog_box)

def getimag(train):  
    img_dir = os.path.join(train) 
    image = Image.open(img_dir)  
    image = image.resize([208, 208])  
    image = np.array(image)
    return image  
  
  
def evaluate(train):  
    image_ori = getimag(train)  
      
    with tf.Graph().as_default():  
        BATCH_SIZE = 1  
        CLASS = 2 
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        image = tf.cast(image_ori, tf.float32)  
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])  
        logit = trainCD.inference(image, BATCH_SIZE, CLASS)  
        logit = tf.nn.softmax(logit)  
  
        logs_train_dir = "D://AI computer system Lab/MyLab/DLcourse/Lab2/New/saveNet/"
        saver = tf.train.Saver()    
        with tf.Session() as sess:                
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
            saver.restore(sess, ckpt.model_checkpoint_path)  
            prediction = sess.run(logit, feed_dict={x: image_ori})
            max_index = np.argmax(prediction)
            return max_index
# 测试
cat_num = 0
dog_num = 0
for i in np.arange(500):
    train = test_box[i]
    max_index = evaluate(train)
    if max_index == 0 and cat_or_dog_box[i] == 1.: # 都是猫
        cat_num += 1.
    elif max_index == 1 and cat_or_dog_box[i] == 0.: #都是狗
        dog_num += 1.
    print("第",i,"次已完成！")
real_accuracy = (cat_num + dog_num)/500.0 
print("准确率是",real_accuracy)