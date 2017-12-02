"""
通过生成的验证码训练模型，其中图片的规格为（60，160，3）
采用的训练模型为cnn，当训练准确率达到98%时停止
"""
#coding: utf-8
import string
from create_captcha import generate_text_and_image
from PIL import Image

import numpy as np
import tensorflow as tf

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
        np.pad(captcha_image, ((98, 98), (48, 48)), 'constant', constant_values=(255, ))
    return captcha_image

MODEL_TEXT, MODEL_IMAGE = generate_text_and_image()
CHAPTCHA_LEN = len(MODEL_TEXT)
CHAR_SET_LEN = len(string.ascii_letters+string.digits)
MODEL_IMAGE = prepare_image(MODEL_IMAGE)
IMAGE_HEIGHT, IMAGE_WIDTH = MODEL_IMAGE.shape

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
    captcha_vector = np.zeros(CHAPTCHA_LEN*CHAR_SET_LEN)
    for i, char in enumerate(captcha_text):
        index = i*CHAR_SET_LEN+char2pos(char)
        captcha_vector[index] = 1
    return captcha_vector

def vec2text(captcha_vector):
    """
    将验证码向量转为字符
    ARGS：
    captcha_vector:验证码字符串
    captcha_len:验证码字符串长度
    RETURN:
    captcha_text:生成的验证码字符
    """

    char_pos = captcha_vector.nonzero()[0]
    captcha_text = ''
    for _, char in enumerate(char_pos):
        char_index = char%CHAPTCHA_LEN
        if char_index < 10:
            captcha_text.join(char_index+ord('0'))
        if char_index < 36:
            captcha_text.join(char_index-10+ord('A'))
        if char_index < 62:
            captcha_text.join(char_index-36+ord('a'))
    return captcha_text

def get_next_batch(batch_size):
    """
    生成用于计算的next_batch
    ARGS:
    next_batch:cnn计算时的步长
    RETURNS：
    batch_x, batch_y:生成的用于计算的向量
    """
    captcha_text, captcha_image = generate_text_and_image()
    captcha_image = prepare_image(captcha_image)
    while captcha_image.shape != MODEL_IMAGE.shape:
        captcha_text, captcha_image = generate_text_and_image()
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, CHAPTCHA_LEN*CHAR_SET_LEN])
    for i in range(batch_size):
        captcha_image = prepare_image(captcha_image)
        batch_x[i, : ] = captcha_image.flatten()/255
        batch_y[i, : ] = text2vec(captcha_text)
    return batch_x, batch_y

def  weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x_image, w):
    return tf.nn.conv2d(x_image, w, strides=[1, 1, 1, 1], padding='SAME')

def max_poop_2x2(x_image):
    return tf.nn.max_pool(x_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def  captcha_cnn():
    x_image = tf.placeholder(tf.float16, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
    x_image = tf.shape(x_image, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    keep_prob = tf.placeholder(tf.float16)

    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
    h_pool1 = max_poop_2x2(h_conv1)

    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
    h_pool2 = max_poop_2x2(h_conv2)

    w_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3)+b_conv3)
    h_pool3 = max_poop_2x2(h_conv3)

    w_fc1 = weight_variable([14*14*128, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 14*14*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024, CHAPTCHA_LEN*CHAR_SET_LEN])
    b_fc2 = bias_variable([1024, CHAPTCHA_LEN*CHAR_SET_LEN])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

    return y_conv

def train_captcha_cnn():

    x_image = tf.placeholder(tf.float16, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
    y_label = tf.placeholder(tf.float16, [None, CHAPTCHA_LEN*CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float16)
    y_conv = captcha_cnn()
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch = get_next_batch(100)
            if step%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x_image:batch[0],\
                                y_label:batch[1], keep_prob:0.75})
                print("step %d , train_accuracy %g"%(step, train_accuracy))
                if train_accuracy > 0.98:
                    saver.save(sess, "captcha_mpodel", global_step=step)
                    break
            train_step.run(feed_dict={x_image:batch[0], y_label:batch[1], keep_prob:0.75})
            step += 1

train_captcha_cnn()
