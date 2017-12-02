"""
通过生成的验证码训练模型，其中图片的规格为（60，160，3）
采用的训练模型为cnn，当训练准确率达到98%时停止
"""
#coding: utf-8
import string
from create_captcha import generate_text_and_image

import numpy as np
import tensorflow as tf

captcha_text, captcha_image = generate_text_and_image()
CHAPTCHA_LEN = len(captcha_text)

def char2pos(word):
    k = ord(word)-48
    if k > 9:
        k = ord(word) - 55
        if k > 35:
            k = ord(word) - 61


def prepare_image(captcha_image):
    if  len(captcha_image.shape()) >2:
        captcha_image = captcha_image.convert("L")
    return captcha_image

def text2vec(captcha_text):
    char_set = string.ascii_letters+string.digits
    captcha_vector = np.zeros(len(captcha_text)*len(char_set))
    for i,c in enumerate(captcha_text):
        index = i*text_len+char2pos(c)
        captcha_vector[i] = 1
    return captcha_vector

def vec2text(captcha_vector):
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

def get_next_batch(batch_size):
    captcha_text, captcha_image=generate_text_and_image()
    image_height, image_weight, _ =captcha_image.shape
    batch_x=np.zeros([batch_size, image_height*image_weight])
    batch_y=np.zeros([batch_size, CHAPTCHA_LEN*len(string.ascii_letters+string.digits)])
    for i in range (batch_size):
        captcha_image=prepare_image(captcha_image)
        batch_x[i, : ]=captcha_image.flatten()/255
        batch_y[i, : ]=text2vec(captcha_text)
    x=tf.placeholder(tf.float16, [None, image_height*image_weight])
    y=tf.placeholder(tf.float16, [None, CHAPTCHA_LEN*len(string.ascii_letters+string.digits)])
    keep_prob=tf.placeholder(tf.float16)



