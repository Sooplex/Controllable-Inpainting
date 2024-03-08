import json
import cv2
import os
from basicsr.utils import img2tensor
import numpy as np

#dataset中保存mask和
class SegDataset():
    def __init__(self, meta_file):
        super(SegDataset, self).__init__()
        self.files = []
        with open(meta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name = line.strip()
                img_path = "/home/pnp/T2I-Adapter-SD/datasets/cropped/"+img_name.rsplit('.', 1)[0]+".png"
                seg_path = "/home/pnp/T2I-Adapter-SD/datasets/seg/"+img_name.rsplit('.', 1)[0]+ '_instance_color_RGB.png'
                txt_path = img_path.rsplit('.', 1)[0] + '.txt'# #NOT USED
                self.files.append({'img_path': img_path, 'seg_path': seg_path, 'txt_path': txt_path})

    def __getitem__(self, idx):
        file = self.files[idx]

        im = cv2.imread(file['img_path'])
        # im = cv2.resize(im, (512, 512))##图像非512×512时需要处理
        seg = cv2.imread(file['seg_path'])  # [:,:,0]
        # seg = cv2.resize(seg, (512, 512))
        seg = img2tensor(seg, bgr2rgb=True, float32=True) / 255.  

        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.
        #prompt处理
        # with open(file['txt_path'], 'r') as fs:
        #     sentence = fs.readline().strip()
        sentence=""#临时prompt

        return {'im': im, 'seg': seg, 'sentence': sentence}

    def __len__(self):
        return len(self.files)
