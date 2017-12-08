#coding:utf-8
import string
from create_captcha import generate_text_and_image

import numpy as np
import tensorflow as tf

LOG_DIR = 'd://log'
MODEL_TEXT, MODEL_IMAGE = generate_text_and_image() #先生成验证码和文字测试模块是否完全
print("验证码图像channel:", MODEL_IMAGE.shape)  # (60, 160, 3)
IMAGE_HEIGHT, IMAGE_WIDTH, _ = MODEL_IMAGE.shape
CAPTCHA_LEN = len(MODEL_TEXT)
CHAR_SET_LEN = len(list(string.ascii_letters + string.digits + '_'))
print("验证码文本最长字符数", CAPTCHA_LEN)

def prepare_image(captcha_image):
    """
    对于验证码进行一系列的准备工作，主要是将验证码由彩色变为黑白
    ARGS：
    captcha_image：待转换的图片
    RETURNS：
    captcha_imgae:转换完毕的图片
    """
    if  len(captcha_image.shape) > 2:
        captcha_image = np.mean(captcha_image, -1)
#    captcha_image = np.pad(captcha_image, ((2, 2), (48, 48)), 'constant', constant_values=(255, ))
    return captcha_image



def char2pos(word):
    """
    将字符变为一个向量中的位置
    ARGS:
    word:待转换的字母
    RETURNS：
    k:转换完毕的字母
    """
    k = ord(word)-48
    if k > 9:
        k = ord(word) - 55
        if k > 35:
            k = ord(word) - 61
    return k

def text2vec(captcha_text):
    """
    将生成的验证码字符转为向量，方便进行运算
    ARGS：
    captcha_text:生成的验证码字符
    RETURN:
    captcha_vector:转换完毕的验证码字符串
    """
    captcha_vector = np.zeros(CAPTCHA_LEN*CHAR_SET_LEN)
    for i, char in enumerate(captcha_text):
        index = i*CHAR_SET_LEN+char2pos(char)
        captcha_vector[index] = 1
    return captcha_vector


def get_next_batch(batch_size=128):
    """
    生成用于计算的next_batch
    ARGS:
    batch_size:cnn计算时的一个batch的大小
    RETURNS：
    batch_x, batch_y:生成的用于计算的向量
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, CAPTCHA_LEN*CHAR_SET_LEN])

    for i in range(batch_size):
        captcha_text, captcha_image = generate_text_and_image()
        captcha_image = prepare_image(captcha_image)

        batch_x[i, : ] = captcha_image.flatten()/255
        batch_y[i, : ] = text2vec(captcha_text)

    return batch_x, batch_y

with tf.name_scope('input'):
    X_IMAGE = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
    Y_LABEL = tf.placeholder(tf.float32, [None, CAPTCHA_LEN*CHAR_SET_LEN])
KEEP_PROB = tf.placeholder(tf.float32) # dropout
X_IMAGE_RESHAPED = tf.reshape(X_IMAGE, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def  weight_variable(shape,layer_name):
    with tf.name_scope(layer_name+'_weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        variable_summaries(initial)
    return tf.Variable(initial)

def bias_variable(shape,layer_name):
    with tf.name_scope(layer_name+'_bias'):
        initial = tf.random_normal(shape = shape)
    return tf.Variable(initial)

def conv2d(x_image, weight_matrix):
    return tf.nn.conv2d(x_image, weight_matrix , strides=[1, 1, 1, 1], padding='SAME')

def max_poop_2x2(x_image):
    return tf.nn.max_pool(x_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def create_layer(layer_name, input_matrix, tensor_shape, bias_shape):
    with tf.name_scope(layer_name):
        w_conv = weight_variable(tensor_shape, layer_name)
        b_conv = bias_variable(bias_shape, layer_name)
        h_conv = tf.nn.relu(conv2d(input_matrix, w_conv)+b_conv)
        h_pool = max_poop_2x2(h_conv)
    return h_pool

def captch_cnn():

    h_pool1 = create_layer('layer_1', X_IMAGE_RESHAPED, [5, 5, 1, 32], [32])
    h_pool2 = create_layer('layer_2', h_pool1, [3, 3, 32, 64], [64])
    h_pool3 = create_layer('layer_3', h_pool2, [3, 3, 64, 128], [128])

    with tf.name_scope('fullt_connected_layer_1'):
        w_fc1 = weight_variable([8*20*128, 1024], 'fullt_connected_layer_1')
        b_fc1 = bias_variable([1024], 'fullt_connected_layer_1')
        h_pool3_flat = tf.reshape(h_pool3, [-1, 8*20*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, KEEP_PROB)

    with tf.name_scope('output_layer'):
        w_fc2 = weight_variable([1024, CAPTCHA_LEN*CHAR_SET_LEN], 'output_layer')
        b_fc2 = bias_variable([CAPTCHA_LEN*CHAR_SET_LEN], 'output_layer')
        y_conv = tf.matmul(h_fc1_drop, w_fc2)+b_fc2
    return y_conv

def train_crack_captcha_cnn():
    y_conv = captch_cnn()
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv,
                                                                               labels=Y_LABEL))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)


    predict = tf.reshape(y_conv, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y_LABEL, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_write = tf.summary.FileWriter(LOG_DIR+'//train', sess.graph)
        test_write = tf.summary.FileWriter(LOG_DIR+'//test')
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(100)
            summary, loss, _ = sess.run([merged, cross_entropy,train_step],
                                 feed_dict={X_IMAGE: batch_x,
                                            Y_LABEL: batch_y, KEEP_PROB: 0.75})
            print(step, loss)
            if step % 10 == 0:
                train_write.add_summary(summary, step)

            if step % 100 == 0 and step != 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc, summary = sess.run( [accuracy,merged], feed_dict={X_IMAGE: batch_x_test,
                                                              Y_LABEL: batch_y_test, KEEP_PROB: 1.})
                print(step, acc)
                test_write.add_summary(summary)
            if step % 600 == 0 and step != 0:
                saver.save(sess, "./model/crack_capcha.model", global_step=step)
                if acc > 0.98:
                    saver.save(sess, "./model/crack_capcha_finish.model", global_step=step)
                    break
            step += 1

train_crack_captcha_cnn()
