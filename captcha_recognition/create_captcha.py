#coding=utf-8
import random
import string
import sys
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

__FONT_PATH = '\\fonts\\arial.ttf'
__CAPTCHA_NUMBER = 6
__CAPTCHA_SIZE = (100, 30)
__BACK_COLOUR = (255, 255, 255)
#字体颜色，默认为蓝色
__FONT_COLOUR = (0, 0, 255)
#干扰线颜色。默认为红色
__LINE_COLOUR = (255, 0, 0)
#是否要加入干扰线
__DRAW_LINE = True
#加入干扰线条数的上下限
__LINE_NUMBER = 5

def __gene_text(__captcha_number = None):
    """生成一个随机的字符串，用于产生验证码，验证码的内容包括了大小写字母和阿拉伯数字
    ARGS：
    __CAPTCHA_NUMBER:字符串的长度
    RETURN:
    captcha_text返回值是一个字符串，用以生成验证码
    """
    if __captcha_number is None:
        captcah_number = __CAPTCHA_NUMBER
    captcha_source = list(string.ascii_letters+string.digits)
    captcha_text = ''.join(random.sample(captcha_source, __CAPTCHA_NUMBER))
    return captcha_text

def __gene_line(draw, width, height,line_colour=None):
    """
    用来绘制干扰线
    ARGS:
    draw是一个传入的画笔
    width,height 是整幅图的宽，即规定了干扰线的范围
    RETURN:
    返回值为空
    """
    if line_colour is None:
        line_colour = __LINE_COLOUR
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill = line_colour)

def gene_code(captcha_size=None, back_colour=None, font_colour=None,line_colour=None, line_number=None):
    """
    用来生成验证码，并将生成的验证码保存。
    ARGS：
    captcha_size:验证码的大小，如果没有输入就为默认值
    back_colour:验证码的背景颜色，如果没有输入，默认为白色
    font_colour:验证码的字的颜色，如果没有输入默认为蓝色
    line_colour:干扰线的颜色默认为红色
    """
    if captcha_size is None:
        captcha_size = __CAPTCHA_SIZE
    if  back_colour is None:
        back_colour = __BACK_COLOUR
    if font_colour is None:
        font_colour = __FONT_COLOUR
    if line_colour is None:
        line_colour = __LINE_COLOUR
    if line_number is None:
        line_number = __LINE_NUMBER

    width, height=captcha_size
    image = Image.new('RGBA', (width, height), back_colour)
    font = ImageFont.truetype(__FONT_PATH, 25) 
    draw = ImageDraw.Draw(image)  
    text = __gene_text() 
    font_width, font_height = font.getsize(text)
    draw.text(((width - font_width) / captcha_number, (height - font_height) / __captcha_number), text, \
            font= font,fill=font_colour) 
    __gene_line(draw, width, height)
    image = image.transform((width+20, height+10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)  #创建扭曲
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE) #滤镜，边界加强
    image.save('idencode.png') 
if __name__ == "__main__":
    gene_code()
