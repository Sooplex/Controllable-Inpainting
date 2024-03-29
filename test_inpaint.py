import os

import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
import numpy as np
import einops

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

# from Resize import (preprocess,apply_paste)

torch.set_grad_enabled(False)


def main():
    parser = get_base_argument_parser()
    
    opt = parser.parse_args()
    which_cond = 'inpaint'
    if opt.outdir is None:
        opt.outdir = f'outputs/test-inpaint1'
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # support two test mode: single image test, and batch test (through a txt file)
    if opt.prompt.endswith('.txt'):#从txt中读取prompt
        assert opt.prompt.endswith('.txt')
        image_paths = []
        prompts = []
        with open(opt.prompt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                prompts.append(line.split('; ')[1])
    else:
        image_paths = [opt.cond_path]
        prompts = [opt.prompt]
    print(image_paths)

    # prepare models
    opt.sampler='ddim'# #choose sampler
    # opt.device ='cuda:0'
    sd_model, sampler = get_sd_models(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    cond_model = None
    

    process_cond_module = getattr(api, 'get_cond_inpaint')#选取api中对应的get_cond函数

    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_image, prompt) in enumerate(zip(image_paths, prompts)):
            # mask= cond_path.rsplit('.', 1)[0]+'_mask000.png'
            mask =opt.mask
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # seed_everything(opt.seed+v_idx+test_idx)
                cond,mask_pixel,img_raw = process_cond_module(opt, cond_image=cond_image, cond_inp_type=opt.cond_inp_type, cond_model=cond_model,mask=mask)#get_cond_inpaint
                # 保存生成的condition
                base_count = len(os.listdir(opt.outdir)) // 2
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond,rgb2bgr=False))
                adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                opt.prompt = prompt
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context,mask_pixel=mask_pixel,img_raw=img_raw)#推理
                # cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), tensor2img(result))
                #组合原图,疑似clamp导致画面不对
                if(mask_pixel is not None and img_raw is not None):
                    mask_pixel_batched= mask_pixel[:,:,None]
                    img_raw_batched = img_raw.copy()
                    tmp=tensor2img(result,rgb2bgr=False)#now rgb
                    cv2.imwrite('./outputs/tmp.png',tmp)
                    # result = (einops.rearrange(result, '1 c h w -> h w c') * 127.5 + 127.5).cpu().numpy().astype(np.float32)#notright
                    result_t = tmp * mask_pixel_batched + img_raw_batched * (1.0 - mask_pixel_batched)#
                    result =result_t.clip(0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), tensor2img(result))


if __name__ == '__main__':
    main()
