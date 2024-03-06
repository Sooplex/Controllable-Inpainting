from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

def Preprocess():
    if det == 'Seg_OFCOCO':
            if not isinstance(preprocessor, OneformerCOCODetector):
                preprocessor = OneformerCOCODetector()
        if det == 'Seg_OFADE20K':
            if not isinstance(preprocessor, OneformerADE20kDetector):
                preprocessor = OneformerADE20kDetector()
        if det == 'Seg_UFADE20K':
            if not isinstance(preprocessor, UniformerDetector):
                preprocessor = UniformerDetector()

        with torch.no_grad():
            input_image = HWC3(input_image)

            if det == 'None':
                detected_map = input_image.copy()
            else:
                detected_map = preprocessor(resize_image(input_image, detect_resolution))
                detected_map = HWC3(detected_map)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)