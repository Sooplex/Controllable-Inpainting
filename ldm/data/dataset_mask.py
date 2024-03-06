import json
import cv2
import os
from basicsr.utils import img2tensor
import numpy as np

#dataset中保存mask和
class MaskDataset():
    def __init__(self, meta_file):
        super(MaskDataset, self).__init__()

        self.files = []
        with open(meta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.strip()
                # img_path = os.path.dirname(meta_file)+"/images/"+img_path# 参考
                img_path = "/home/pnp/lama-main/output/"+img_path.rsplit('.', 1)[0]+"_crop000.png"
                mask_path = img_path.rsplit('.', 1)[0] + '_mask000.png'
                txt_path = img_path.rsplit('.', 1)[0] + '.txt'# #NOT USED
                self.files.append({'img_path': img_path, 'mask_path': mask_path, 'txt_path': txt_path})

    def __getitem__(self, idx):
        file = self.files[idx]

        im = cv2.imread(file['img_path'])
        im = cv2.resize(im, (512, 512))
        mask = cv2.imread(file['mask_path'])  # [:,:,0]
        # 以同样的方式将mask和im整合成cond
        mask = cv2.resize(mask, (512, 512))
        mask_blur=0.5#训练过程中也需要设置
        mask_pixel = cv2.resize(mask[:, :, 0], (512,512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0#压缩mask的维度
        mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)
        #检测含mask区域
        detected_map = im.copy()
        detected_map[mask_pixel > 0.5] = - 255.0 # blur过的mask_pixel>0.5的所有位置
        cond = img2tensor(detected_map,bgr2rgb=True,float32=True)/ 255.

        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.#这样子转换居然不会影响im的赋值
        #prompt处理
        # with open(file['txt_path'], 'r') as fs:
        #     sentence = fs.readline().strip()
        sentence=""#临时prompt

        return {'im': im, 'im_masked': cond, 'sentence': sentence}

    def __len__(self):
        return len(self.files)
