"生成用于训练和测试的验证码数据集"
import random
import string
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_text_and_image(char_set=None, text_size=None):
    """
    从字符集中随机选择字符生成一个默认长度为6的字符串，并生成验证码图片
    Args:
    charSet:字符集，默认为数字，小写字母和大写字母。
    text_size:验证码长度，默认为6
    Return:
    captcha_text:随机生成的验证码的文字
    captcha_picture:与captcha_text相匹配的验证码图片
    """
    if char_set is None:
        char_set = list(string.ascii_letters+string.digits)
    if text_size is None:
        text_size = 6
    captcha_text = ''
    for _ in range(text_size):
        random_char = random.choice(char_set)
        captcha_text = captcha_text+random_char

    captcha = ImageCaptcha().generate(captcha_text)

    captcha_image = np.array(Image.open(captcha))
    ImageCaptcha().write(captcha_text, captcha_text + '.jpg')
    return captcha_text, captcha_image

def main():
    "用于测试,讲生成的验证码及其文字显示在屏幕上，文字在验证码上方"
    text, image = generate_text_and_image()
    plt.title(text)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
