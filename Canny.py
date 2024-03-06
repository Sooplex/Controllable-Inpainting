import os
import cv2
from basicsr.utils import img2tensor,tensor2img
from ldm.util import resize_numpy_image


import sys
os.chdir(sys.path[0])
outdir='/home/pnp/T2I-Adapter-SD/datasets/canny/images'
def Process_canny(cond_image):
    name=cond_image.rsplit('.', 1)[0]

    canny = cv2.imread(cond_image)
    # canny = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    canny = resize_numpy_image(canny)
    cv2.imwrite(os.path.join(f'{name}.png'), canny) #存resize的原图保证数据一致
    canny = cv2.Canny(canny, 100, 200)[..., None]
    canny = img2tensor(canny).unsqueeze(0) / 255.
    
    # cv2.imwrite(os.path.join(sys.path[0], f'{name}_canny.png'), tensor2img(canny))
    cv2.imwrite(os.path.join(f'{name}_canny.png'), tensor2img(canny))#cond_image同目录下放canny
    return canny
def main():
    # meta_file='/home/pnp/T2I-Adapter-SD/datasets/canny/canny_RS.txt'
    # with open(meta_file, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             cond_image = line.strip()
    #             cond_image="/home/pnp/T2I-Adapter-SD/datasets/canny/images/"+cond_image#直接给出full path
    #             Process_canny(cond_image)
    Process_canny("/home/pnp/T2I-Adapter-SD/datasets/Forest.png")
if __name__ == '__main__':
    main()

    