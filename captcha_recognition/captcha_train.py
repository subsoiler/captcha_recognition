"""
通过生成的验证码训练模型，其中图片的规格为（60，160，3）
采用的训练模型为cnn，当训练准确率达到98%时停止
"""
#coding: utf-8
import string
from create_captcha import generate_text_and_image

import numpy as np
import tensorflow as tf

MODEL_TEXT, MODEL_IMAGE = generate_text_and_image()
CHAPTCHA_LEN = len(MODEL_TEXT)

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

def prepare_image(captcha_image):
    """
    对于验证码进行一系列的准备工作，主要是将验证码由彩色变为黑白
    ARGS：
    captcha_image：待转换的图片
    RETURNS：
    captcha_imgae:转换完毕的图片
    """
    if  len(captcha_image.shape()) >2:
        captcha_image = captcha_image.convert("L")
    return captcha_image

def text2vec(captcha_text):
    """
    将生成的验证码字符转为向量，方便进行运算
    ARGS：
    captcha_text:生成的验证码字符
    RETURN:
    captcha_vector:转换完毕的验证码字符串
    """
    char_set = string.ascii_letters+string.digits
    captcha_vector = np.zeros(len(captcha_text)*len(char_set))
    for i,c in enumerate(captcha_text):
        index = i*text_len+char2pos(c)
        captcha_vector[i] = 1
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
    for _, c in enumerate(char_pos):
        char_index=c%CHAPTCHA_LEN
        if char_index <10:
            captcha_text.join(char_index+ord('0'))
        if char_index<36:
            captcha_text.join(char_index-10+ord('A'))
        if char_index <62:
            captcha_text.join(char_index-36+ord('a'))
    return captcha_text

def get_next_batch(batch_size, captcha_len):
    """
    生成用于计算的next_batch
    ARGS:
    next_batch:cnn计算时的步长
    RETURNS：
    batch_x, batch_y:生成的用于计算的向量
    """
    captcha_text, captcha_image=generate_text_and_image()
    while captcha_image.shape!=MODEL_IMAGE.shape:
        captcha_text, captcha_image=generate_text_and_image()
    image_height, image_weight, _ =captcha_image.shape
    batch_x=np.zeros([batch_size, image_height*image_weight])
    batch_y=np.zeros([batch_size, CHAPTCHA_LEN*len(string.ascii_letters+string.digits)])
    for i in range (batch_size):
        captcha_image=prepare_image(captcha_image)
        batch_x[i, : ]=captcha_image.flatten()/255
        batch_y[i, : ]=text2vec(captcha_text)
    return batch_x, batch_y

def  weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

