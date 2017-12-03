#coding:utf-8
import string
from create_captcha import generate_text_and_image

import numpy as np
import tensorflow as tf

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


X_IMAGE = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y_LABEL = tf.placeholder(tf.float32, [None, CAPTCHA_LEN*CHAR_SET_LEN])
KEEP_PROB = tf.placeholder(tf.float32) # dropout

def  weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x_image, weight_matrix):
    return tf.nn.conv2d(x_image, weight_matrix, strides=[1, 1, 1, 1], padding='SAME')

def max_poop_2x2(x_image):
    return tf.nn.max_pool(x_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def captch_cnn():
    x_image = tf.reshape(X_IMAGE, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

	# 3 conv layer
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
    h_pool1 = max_poop_2x2(h_conv1)

    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
    h_pool2 = max_poop_2x2(h_conv2)

    w_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3)+b_conv3)
    h_pool3 = max_poop_2x2(h_conv3)

    w_fc1 = weight_variable([8*20*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*20*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, KEEP_PROB)

    w_fc2 = weight_variable([1024, CAPTCHA_LEN*CHAR_SET_LEN])
    b_fc2 = bias_variable([CAPTCHA_LEN*CHAR_SET_LEN])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)
    return y_conv

def train_crack_captcha_cnn():
    y_conv = captch_cnn()
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv,
                                                                           labels=Y_LABEL))

    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

    predict = tf.reshape(y_conv, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y_LABEL, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([train_step, cross_entropy],
                                feed_dict={X_IMAGE: batch_x, Y_LABEL: batch_y, KEEP_PROB: 0.75})
            print(step, loss_)

            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X_IMAGE: batch_x_test,
                                                    Y_LABEL: batch_y_test, KEEP_PROB: 1.})
                print(step, acc)
				# 如果准确率大于98%,保存模型,完成训练
                if acc > 0.98:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break
            step += 1

train_crack_captcha_cnn()
