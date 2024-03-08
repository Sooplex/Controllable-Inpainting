import json
import cv2
import os
from basicsr.utils import img2tensor


class dataset_iSAID_mask_color():
    ##coco version
    # def __init__(self, path_json, root_path_im, root_path_mask, image_size):
    #     super(dataset_iSAID_mask_color, self).__init__()
    #     with open(path_json, 'r', encoding='utf-8') as fp:
    #         data = json.load(fp)
    #     data = data['annotations']
    #     self.files = []
    #     self.root_path_im = root_path_im
    #     self.root_path_mask = root_path_mask
    #     for file in data:
    #         name = "%012d.png" % file['image_id']
    #         self.files.append({'name': name, 'sentence': file['caption']})
    def __init__(self, meta_file):
        super(dataset_iSAID_mask_color, self).__init__()

        
        with open(meta_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        for file in data:
            # img_path = os.path.dirname(meta_file)+"/images/"+img_path# 自己做的.txt给出的路径不完全，就补一补
            img_path = "/home/pnp/T2I-Adapter-SD/datasets/cropped/"+img_path
            txt_path = img_path.rsplit('.', 1)[0] + '.txt'
            # self.files.append({'name': name, 'sentence': file['caption']})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png', '.jpg')))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name))  # [:,:,0]
        mask = cv2.resize(mask, (512, 512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im': im, 'mask': mask, 'sentence': sentence}

    def __len__(self):
        return len(self.files)
