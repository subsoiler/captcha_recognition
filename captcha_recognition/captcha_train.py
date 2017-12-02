#coding: utf-8
from create_captcha import generate_text_and_image
from PIL import image

import numpy as np
import tensorflow as tf
import string

captcha_text, captcha_image=generate_text_and_image()
print(captcha_image.shape())
CHAPTCHA_LEN=len(captcha_text)

def char2pos(c):
    k=ord(c)-48
    if k > 9:
        k = ord(c) - 55
        if k > 35:
            k = ord(c) - 61


def prepare_image(captcha_image):
    if  len(captcha_image.shape()) >2:
        captcha_image=captcha_image.convert("L")
    np.pad(image,((2,3), (2,2 )),'constant',constant_value = (255, ))

def text2vec(captcha_text):
    char_set = string.ascii_letters+string.digits
    captcha_vector = np.zeros(len(captcha_text)*char_set)
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
    batch_x=np.zeros([batch_size, ])


