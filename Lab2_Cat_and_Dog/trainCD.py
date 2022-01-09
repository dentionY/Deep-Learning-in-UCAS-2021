#import tensorflow as tf
import tensorflow.compat.v1 as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_v2_behavior()
tf.reset_default_graph()
import numpy as np
from matplotlib import pyplot as plt 


def get_files(filedir):
    cat_imag = []
    dog_imag = []
    cat_lab  = []
    dog_lab  = []
    for file in os.listdir(filedir): #os.listdir 将文件名分解
        name = file.split(sep = '.')  # 此时 name 按照 [cat or dog, 数字, jpg] 分开
        if name[0] == 'cat':
            cat_imag.append(filedir + file) # 这一步是为了将train文件夹内所有的猫图片以path的形式保存
            cat_lab.append(0)
        elif name[0] == 'dog':
            dog_imag.append(filedir + file)
            dog_lab.append(1)
            # 根据要求，随机选取2500张train中的图片
        imag = np.hstack((cat_imag, dog_imag))
        lab  = np.hstack((cat_lab, dog_lab))
    imag_lab = np.array([imag,lab])
    imag_lab = imag_lab.transpose()
    # 打乱imag_lab
    np.random.shuffle(imag_lab)
    imag_rand_list = list(imag_lab[0:10000,0])
    lab_rand_list  = list(imag_lab[0:10000,1])
    lab_rand_list = [int(i) for i in lab_rand_list]
    # 选取2000张作为训练集
    imag_rand_list_train = imag_rand_list[0:9500]
    lab_rand_list_train = lab_rand_list[0:9500]
    # 选取500张作为测试集
    imag_rand_list_test = imag_rand_list[9500:10000]
    lab_rand_list_test = lab_rand_list[9500:10000]
    for i in range(500):
        test_sample_file = open("D:\\AI computer system Lab\\MyLab\\DLcourse\\Lab2\\New\\test_sample.txt",'a')
        print(imag_rand_list_test[i],file = test_sample_file)
        test_sample_file.close()
        test_label_file = open("D:\\AI computer system Lab\\MyLab\\DLcourse\\Lab2\\New\\test_label.txt",'a')
        if float(lab_rand_list_test[i]) == 1.:
            print('dog',file = test_label_file)
        elif float(lab_rand_list_test[i]) == 0.:
            print('cat',file = test_label_file)
        test_label_file.close()
    return imag_rand_list_train,lab_rand_list_train,imag_rand_list_test,lab_rand_list_test


#train_dir = "D://AI computer system Lab/MyLab/DLcourse/kaggle/train/"
#image_list_train,label_list_train,image_list_test,label_list_test = get_files(train_dir)


def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels =3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],batch_size = batch_size, num_threads = 64, capacity = capacity)  
    label_batch = tf.reshape(label_batch , [batch_size])
    image_batch = tf.cast(image_batch,tf.float32)
    return  image_batch, label_batch

# 结构
# conv1   卷积层 1
# pooling1  池化层 1
# conv2  卷积层 2
# pooling2  池化层 2
# conv3  卷积层 3
# pooling3 池化层 3
# conv4  卷积层 4
# pooling4  池化层 4
# local3 全连接层 1
# local4 全连接层 2
# softmax 全连接层 3
def inference(images, batch_size, n_classes):  
    weights = tf.Variable(tf.truncated_normal([3, 3, 3, 16],stddev=0.1))
    biases = tf.Variable(tf.constant(0.1,shape = [16]))
    conv = tf.nn.conv2d(images, weights, strides=[1, 2, 2, 1], padding='SAME')  
    pre_activation = tf.nn.bias_add(conv, biases)  
    conv1 = tf.nn.relu(pre_activation)  
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  
    norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  
  
     
    weights = tf.Variable(tf.truncated_normal([3, 3, 16, 32],stddev=0.1))
    biases = tf.Variable(tf.constant(0.1,shape = [32]))
    conv = tf.nn.conv2d(norm1, weights, strides=[1, 2, 2, 1], padding='SAME')  
    pre_activation = tf.nn.bias_add(conv, biases)  
    conv2 = tf.nn.relu(pre_activation)   
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')   
    norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  
    

    weights = tf.Variable(tf.truncated_normal([3, 3, 32, 32],stddev=0.1))
    biases = tf.Variable(tf.constant(0.1,shape = [32]))
    conv = tf.nn.conv2d(norm2, weights, strides=[1, 2, 2, 1], padding='SAME')  
    pre_activation = tf.nn.bias_add(conv, biases)  
    conv3 = tf.nn.relu(pre_activation)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')   
    norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)   
    

    weights = tf.Variable(tf.truncated_normal([3, 3, 32, 32],stddev=0.1))
    biases = tf.Variable(tf.constant(0.1,shape = [32])) 
    conv = tf.nn.conv2d(norm3, weights, strides=[1, 2, 2, 1], padding='SAME')  
    pre_activation = tf.nn.bias_add(conv, biases)  
    conv4 = tf.nn.relu(pre_activation)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')   
    norm4 = tf.nn.lrn(pool4, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

  
    reshape = tf.reshape(norm4, shape=[batch_size, -1])  
    dim = reshape.get_shape()[1].value 
    weights = tf.Variable(tf.truncated_normal([dim, 1024],stddev=0.005))
    biases = tf.Variable(tf.constant(0.1,shape = [1024]))  
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)  

    weights = tf.Variable(tf.truncated_normal([1024,512],stddev=0.005))
    biases = tf.Variable(tf.constant(0.1,shape = [512]))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)  
  
    weights = tf.Variable(tf.truncated_normal([512, n_classes],stddev=0.005))
    biases = tf.Variable(tf.constant(0.1,shape = [n_classes]))  

    softmax_linear = tf.add(tf.matmul(local4, weights), biases) 
  
    return softmax_linear  


STEP = 25000 # 训练的步数

train_dir = "D://AI computer system Lab/MyLab/DLcourse/kaggle/train/"  
image_list_train,label_list_train,image_list_test,label_list_test = get_files(train_dir)
# 打印训练集中选取用于测试的猫和狗的路径
#test_sample_file = open("D:\\AI computer system Lab\\MyLab\\DLcourse\\Lab2\\New\\test_sample.txt",'w+')
#print('There are 500 random samples of cats and dogs!',file = test_sample_file)
#test_sample_file.close()
#test_sample_file = open("D:\\AI computer system Lab\\MyLab\\DLcourse\\Lab2\\test_sample.txt",'a')
#print('Cat : ',sample_cat_num_test,', Dog : ',sample_dog_num_test,file = test_sample_file)
#test_sample_file.close() 
step_box = [] 
train_loss_box = []
train_accu_box = [] 
def training():  
    logs_train_dir = "D://AI computer system Lab/MyLab/DLcourse/Lab2/New/saveNet/" 
    train_batch, train_label_batch = get_batch(image_list_train, label_list_train, 208, 208, 32, 256)
    train_logits = inference(train_batch, 32, 2)  
    train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=train_label_batch)) 
    train_op = tf.train.AdamOptimizer(0.0001).minimize(train_loss)
    train_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(train_logits, train_label_batch, 1) , tf.float16))  
    sess = tf.Session()  
    saver = tf.train.Saver()  
      
    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
 
    for step in np.arange(STEP):   
        tra_op, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])  
                 
        step_box.append(step)
        train_loss_box.append(tra_loss)
        train_accu_box.append(tra_acc)  
        print('Step',step, 'train loss = ',tra_loss, 'train accuracy = ' , tra_acc*100.0)  
               
        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
        saver.save(sess, checkpoint_path, global_step=step)  
                    
    coord.request_stop()
    coord.join(threads)  
    sess.close()  

if __name__ == "__main__":
    training()
    plt.figure(1)
    plt.title("Training loss--Training iter") 
    plt.xlabel("Training iter") 
    plt.ylabel("Training loss") 
    plt.plot(step_box,train_loss_box) 
    plt.show()
    plt.figure(2)
    plt.title("Training acc--Training iter") 
    plt.xlabel("Training iter") 
    plt.ylabel("Training acc") 
    plt.plot(step_box,train_accu_box) 
    plt.show()    