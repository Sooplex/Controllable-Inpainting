import json
import cv2
import os
from basicsr.utils import img2tensor
import numpy as np


class CannyDataset():
    def __init__(self, meta_file):
        super(CannyDataset, self).__init__()

        self.files = []
        with open(meta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.strip()
                # img_path = os.path.dirname(meta_file)+"/images/"+img_path# 自己做的.txt给出的路径不完全，就补一补
                img_path = "/home/pnp/T2I-Adapter-SD/datasets/cropped/"+img_path
                canny_img_path = img_path.rsplit('.', 1)[0] + '_canny.png'
                txt_path = img_path.rsplit('.', 1)[0] + '.txt'
                self.files.append({'img_path': img_path, 'canny_img_path': canny_img_path, 'txt_path': txt_path})

    def __getitem__(self, idx):
        file = self.files[idx]

        im = cv2.imread(file['img_path'])
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.#这样子转换居然不会影响im的赋值

        canny = cv2.imread(file['canny_img_path'])  # [:,:,0]
        ##需要把canny输入调成64channels
        # canny = img2tensor(canny, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.
        canny = img2tensor(canny, bgr2rgb=True, float32=True)[0:1] / 255.      
        # with open(file['txt_path'], 'r') as fs:
        #     sentence = fs.readline().strip()
        sentence="Satellite image"#txt也没配图，就直接随便搞一个把

        return {'im': im, 'canny': canny, 'sentence': sentence}

    def __len__(self):
        return len(self.files)
