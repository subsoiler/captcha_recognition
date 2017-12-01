#coding=utf-8
import random
import string
import sys
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

#字体的位置，不同版本的系统会有不同
__FONT_PATH = '\\fonts\\arial.ttf'
__CAPTCHA_NUMBER = 4
__CAPTCHA_SIZE = (100, 30)
__BACK_COLOUR = (255,255,255)
#字体颜色，默认为蓝色
__FONG_COLOUR = (0,0,255)
#干扰线颜色。默认为红色
__LINE_COLOUR = (255,0,0)
#是否要加入干扰线
__DRAW_LINE = True
#加入干扰线条数的上下限
__LINE_NUMBER = (1, 5)

def __gene_text():
    """生成一个随机的字符串，用于产生验证码，验证码的内容包括了大小写字母和阿拉伯数字
    ARGS：
    __CAPTCHA_NUMBER:字符串的长度
    RETURN:
    返回值是一个字符串，用以生成验证码
    """
    source = list(string.ascii_letters+string.digits)
    for index in range(0,10):
        source.append(str(index))
    return ''.join(random.sample(source,__CAPTCHA_NUMBER))
#用来绘制干扰线
def __gene_line(draw, width, height):
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill = __LINE_COLOUR)

#生成验证码
def gene_code():
    width,height = __CAPTCHA_SIZE#宽和高
    image = Image.new('RGBA',(width,height),__BACK_COLOUR) #创建图片
    font = ImageFont.truetype(__FONT_PATH,25) #验证码的字体
    draw = ImageDraw.Draw(image)  #创建画笔
    text = __gene_text() #生成字符串
    font_width, font_height = font.getsize(text)
    draw.text(((width - font_width) / __CAPTCHA_NUMBER, (height - font_height) / __CAPTCHA_NUMBER),text,\
            font= font,fill=__FONG_COLOUR) #填充字符串
    if __DRAW_LINE:
        __gene_line(draw,width,height)
    image = image.transform((width+20,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE) #滤镜，边界加强
    image.save('idencode.png') #保存验证码图片
if __name__ == "__main__":
    gene_code()
