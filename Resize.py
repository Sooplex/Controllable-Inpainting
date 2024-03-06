import cv2
import numpy as np
import os

import sys
os.chdir(sys.path[0])
# pic_path = '.jpg' # 分割的图片的位置
# outdir = './datasets/cropped' # 分割后的图片保存的文件夹
outdir ='datasets/seg'
#将单张图片切成等比，缩放成512*512
def Resize_image(image):
    img = cv2.imread(image)
    x, y = img.shape[0:2]
    s = x if x<y else y
    # cropped = img[0:512, 0:512]
    cropped = img[0:s, 0:s]
    cropped=cv2.resize(cropped,(512,512))#必须赋值一下
    name=image.rsplit('/',1)[-1]
    if not os.path.exists(outdir):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(outdir)
    cv2.imwrite(outdir+f'/{name}', cropped)
#对于分辨率太高，细节足够丰富的大图进行
def Crop_image(img):
    
    if not os.path.exists(outdir):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(outdir)
    #要分割后的尺寸
    cut_width = 512
    cut_length = 512
    # 读取要分割的图片，以及其尺寸等数据
    picture = cv2.imread(img)
    (width, length, depth) = picture.shape
    # 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
    # 计算可以划分的横纵的个数
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    # for循环迭代生成
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = picture[i*cut_width : (i+1)*cut_width, j*cut_length : (j+1)*cut_length, :]      
            result_path = outdir + '{}_{}.jpg'.format(i+1, j+1)
            cv2.imwrite(result_path, pic)
    
def main():
    meta_file='/home/pnp/T2I-Adapter-SD/datasets/canny/canny_RS.txt'
    with open(meta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image = line.strip()
                # canny="/home/pnp/T2I-Adapter-SD/datasets/canny/images_bak/"+image.split('.',1)[0]+'_canny.png'#直接给出full path
                # Resize_image(canny)
                # img="/home/pnp/T2I-Adapter-SD/datasets/canny/images_bak/"+image
                # Resize_image(img)
                cond_image="/home/pnp/T2I-Adapter-SD/datasets/Smask/images/"+image.rsplit('.',1)[0]+'_instance_color_RGB.png'
                Resize_image(cond_image)
    # for img in os.listdir('datasets/cropped'):
    #     Resize_image("/home/pnp/T2I-Adapter-SD/datasets/cropped/"+img)
    # Resize_image('/home/pnp/T2I-Adapter-SD/datasets/canny/images_bak/P0000.png')#测试一下单张图片可否
    # Resize_image('/home/pnp/T2I-Adapter-SD/datasets/Smask/images/P0821_instance_color_RGB.png')
                
if __name__ == '__main__':
    main()
