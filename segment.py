# from share import *
import config

import argparse
import warnings
import os
import cv2
import einops
import re
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image, ImageColor

from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from functools import partial
from deepfloyd_if.modules.stage_II import IFStageII

grad_channels = 0
masks = []
rgbs = []
sam = 0
predictor = 0
mask_generator = 0
masklist =[]

stage_2 = 0

MAX_COLORS = 128

def load(model_load_path, sam_path):
    global sam
    global predictor
    global mask_generator
    #"/home/faariss2/PaletteControl/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to(device='cuda:0')
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(model=sam)


    #options
    IMAGE_SIZE = 256
    aux_channels = 7
    doCN = True
    force_aux_8 = False
    mpath = 'IF-II-M-v1.0' #small version of DeepFloyd
    #"/home/faariss2/G.pt"

    model_kwargs = {'doCN': doCN, 'aux_ch': aux_channels + (8 - aux_channels % 8) % 8 if force_aux_8 else aux_channels, 'attention_resolutions': '32,16'}

    global stage_2
    stage_2 = IFStageII(mpath, device='cuda:0', filename=model_load_path, model_kwargs=model_kwargs)

    def _setDtype(stage_2):
        stage_2.model.dtype = torch.float32 #tested on float32 mixed precision
        stage_2.model.precision = '32' 
        if doCN:
            stage_2.model.control_model.dtype = stage_2.model.dtype
            stage_2.model.control_model.precision = stage_2.model.precision
        for name, p in stage_2.model.named_parameters():
            p.data = p.type(stage_2.model.dtype)


    _setDtype(stage_2)

    for name, p in stage_2.model.named_parameters():
        p.requires_grad = False

    stage_2.model.eval()

    return


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    #round to closest number divisible by 16
    H = int(np.round(H / 16.0)) * 16
    W = int(np.round(W / 16.0)) * 16
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def center_crop(arr, new_height, new_width):
    #center crops image to be new_height new_width
    h, w = arr.shape[:2]
    start_y = max(0, (h - new_height) // 2)
    start_x = max(0, (w - new_width) // 2)
    return arr[start_y:start_y+new_height, start_x:start_x+new_width]


def show_masks(imag):
    #Runs SAM and returns image with masks
    width, height = imag.size
    #upsample to remove bad edges
    # upscaled = imag.resize((int(1*width), int(1*height)))

    if mask_generator == 0:
        raise ValueError("Models Haven't been Loaded")
    arr = np.asarray(imag)
    anns = mask_generator.generate(arr)
    global masks
    masks = anns
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    
    im = Image.fromarray((img*255).astype(np.uint8))



    imag.paste(im, mask=im)
    return imag





def app_options():
    #Gradio arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", '-addr', type=str, default="0.0.0.0")
    parser.add_argument("--server_port", '-port', type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--not_show_error", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_text_manipulation", '-manipulate', action="store_true")
    parser.add_argument('--model_load_path', required=False, default=None,
      help='pretrained model, None means train from scratch.')
    # parser.add('--data_mode', required=False, default='G',
    #   choices=('L', 'G', 'T'), help='flags for dataset: "L" for we will condition on Luminance. "G" for gradient. "T" for Thresholded gradient.')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction)
    return parser.parse_args()



def getGradResize(input_image, image_resolution, mode):
    if input_image == None:
         return [None, None]
    masklist.clear()
    rgbs.clear()
    img = resize_image(np.asarray(input_image), image_resolution)

    #must be divisible by 8
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
   
    gradients = np.gradient(gray_image)

    resizedIm = Image.fromarray(img)
    
    global grad_channels
    if mode == 'Gradient':
            grad_channels = np.stack([(g + 255) / (2 * 255.) for g in gradients], axis=-1)
    elif mode == 'Luminance':
            luminance = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0] / 255.0
            grad_channels = np.stack([luminance, luminance], axis=-1)
    elif mode == 'Threshold':
            grad_channels = np.stack([np.greater(np.abs(g), 8).astype(np.float32) for g in gradients], axis=-1)
    else:
        raise ValueError("Invalid value provided for mode")

    return [resizedIm, resizedIm]


def process(input_image, textureOption):
    with torch.no_grad():
        effective_batch_size = 1
        steps = "super27"
        aug_level =  0.0
        support_noise_less_qsample_steps = 0
        dynamic_thresholding_p = 0.95
        dynamic_thresholding_c =  1.0
        sample_loop = 'ddpm'
        patchImage = np.asarray(input_image)

        height, width = patchImage.shape[:2]
        
        detected_map = np.ones((height, width, 7), dtype=np.float32) #7 channels in the conditioning
        

        texture_channel = np.ones((height, width))
        num_colors = 8
        color_indicator = np.full((height, width), num_colors / (2 * MAX_COLORS))

        detected_map = np.ones((height, width, 7))

        detected_map[:, :, :3] = patchImage
        detected_map[:, :, 3:5] = grad_channels #(grad[0] +255)/2 #
        global masklist
        global rgbs
        for i in range(len(rgbs)):
            mask = masklist[i]
            rgb=rgbs[i]
            #The colors are already changed since we change input_image in fillColor
            detected_map[:, :, 0][mask] = rgb[0]
            detected_map[:, :, 1][mask] = rgb[1]
            detected_map[:, :, 2][mask] = rgb[2]
            color_indicator[mask] = 1 / (2 * MAX_COLORS)
            color_indicator[~mask] = 1.
            if textureOption:
                texture_channel[mask] = 0
        detected_map[:, :, 3][texture_channel == 0] = 0
        detected_map[:, :, 4][texture_channel == 0] = 0

        detected_map[:, :, 6] = texture_channel

        detected_map[:, :, 5] = color_indicator
    
        

        detected_map[:, :, :3] /= 255.0

        color = np.copy(detected_map[:, :, :3])

        if opt.fp16:
            control = torch.from_numpy(detected_map.copy()).half().cuda()
            color = torch.from_numpy(color.copy()).half().cuda()
        else:
            control = torch.from_numpy(detected_map.copy()).float().cuda()
            color = torch.from_numpy(color.copy()).float().cuda()


        # control = np.transpose(detected_map, (2, 0, 1))
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        color = torch.stack([color for _ in range(1)], dim=0)
        color = einops.rearrange(color, 'b h w c -> b c h w').clone()
        # text_prompts = torch.from_numpy(np.load('empty_prompt_1_77_4096.npz', allow_pickle=True)[
        #                         'arr']).to('cuda:0').repeat(1, 1, 1)
        text_prompts = torch.from_numpy(np.load('empty_prompt_1_77_4096.npz', allow_pickle=True)[
                                'arr']).to('cuda:0').repeat(effective_batch_size, 1, 1)

        with torch.autocast("cuda", dtype=torch.float16): 
            out, metadata = stage_2.embeddings_to_image(sample_timestep_respacing=str(steps), 
                low_res=2*color-1, support_noise=2*color-1,
                support_noise_less_qsample_steps=support_noise_less_qsample_steps, 
                seed=None, t5_embs=text_prompts[0:1, ...], hint=2*control-1, 
                aug_level=aug_level, sample_loop=sample_loop, 
                dynamic_thresholding_p=dynamic_thresholding_p,dynamic_thresholding_c=dynamic_thresholding_c)
            
        out = (out + 1)/2


        
        # if seed == -1:
        #     seed = random.randint(0, 65535)
        # seed_everything(seed)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=False)

        # cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([', ' + a_prompt] * num_samples)]}
        # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        # #change all 512s back

      

        x_samples = [(255*out.squeeze().cpu().numpy().transpose(1,2,0)).astype(np.uint8)]

        results = [x_samples[i] for i in range(1)]
        # Image.fromarray(results[0]-np.asarray(input_image)).save("res.jpg")


    return results


def fillColor(palette, color, input_image, event: gr.SelectData):
    width, height = palette.size
    if event is not None:
        y, x = event.index[0], event.index[1]
        area = 9223372036854775807
        if x is not None and y is not None:
            input_point = np.array([[y, x]])
            input_label = np.array([1])
            
            #float bug rgba(159.29002075195314, 65.32288131713868, 65.32288131713868, 1)'
            numbers = re.findall(r"[\d.]+", color)
            r, g, b = tuple(int(round(float(n))) for n in numbers[:3])
            width, height = input_image.size
            global masks
            background = np.ones_like(np.asarray(palette)[:, :, 1], dtype=bool) 
            notFound = True
            for m in masks:
                m['segmentation'] = center_crop(m['segmentation'], height, width)
                background[m['segmentation']] = 0
                if (m['segmentation'][x, y] and m['area'] < area):
                    # print("hel")
                    best= m['segmentation']
                    area = m['area']
                    notFound = False
            if notFound:
                background[:447, :] = False
                best = (background) #& ((np.abs(xyz[:,:,0]-xyz[x, y, 0]))<50))
            img = np.array(input_image)
            img[best] = [r,g,b]

            pall = np.array(palette)
            pall[best] = [r,g,b]
            global rgbs
            rgbs.append([r,g,b])
            global masklist
            masklist.append(best)
            return Image.fromarray(img), Image.fromarray(pall)
        return ("", "")
    return ("", "")


# sampler_list = get_sampler_list()
# scheduler_list = get_noise_schedulers()
opt = app_options()

img_with_mask = partial(gr.Image, type="pil", height=300, interactive=True, show_label=True)
# warnings.simplefilter('error')

with gr.Blocks(
        title="Colorize Diffusion",
        theme=gr.themes.Soft(),
        elem_id="main-interface",
        analytics_enabled=False
) as block:
    with gr.Row(elem_id="content-row", equal_height=False, variant="panel"):
                with gr.Column(scale=1):
                    image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1088, value=256, step=64)
                    mode = gr.Radio(["Luminance", "Gradient", "Threshold"], label="Mode")
                    model_path = gr.Textbox(value="/home/faariss2/G.pt", label="model_path")
                    sam_path = gr.Textbox(value="/home/faariss2/PaletteControl/sam_vit_h_4b8939.pth", label="sam_path")
                    load_models = gr.Button("Load Models", variant="secondary", size="sm")
                    input_image = gr.Image(label = "Source Image", sources='upload', type="pil")
                    
                    
                with gr.Column(scale=6):
                    control = gr.Image(label = "Control Image", type="pil")
                    picker = gr.ColorPicker()
                    get_mask = gr.Button(value="Get Mask")
                    palette = gr.Image(label = "Click a Segment",sources='upload', type="pil", scale=3)
                    # with gr.Accordion("Advanced options", open=False):
                        # steps = gr.Textbox(value="super27", label="steps")
                        # aug_level = gr.Number(value = 0.0, label="aug_level")
                        # support_noise_less_qsample_steps = gr.Number(value = 0, label="support_noise_less_qsample_steps")
                        # dynamic_thresholding_p = gr.Number(value = 0.95,label="dynamic_thresholding_p")
                        # dynamic_thresholding_c = gr.Number(value = 1.0,label="dynamic_thresholding_p")
                        # sample_loop = gr.Textbox(value='ddpm', label="sample_loop")
                   

                    textureOption = gr.Checkbox(label="Texture Dropout", info="Check to Dropout Texture Layer")
                    run_button = gr.Button("🚀 Generate", variant="primary", size="lg")



                with gr.Column(scale=4):
                    result = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True)


            
  

    palette.select(fn=fillColor, inputs=[palette, picker, control], outputs=[control, palette])   
    run_button.click(fn=process, inputs=[control, textureOption], outputs=[result])
    get_mask.click(fn=show_masks, inputs=[palette], outputs=[palette])
    input_image.input(fn=getGradResize, inputs=[input_image, image_resolution, mode], outputs=[control, palette])
    load_models.click(fn=load, inputs=[model_path, sam_path])
    
    block.launch(
            server_name=opt.server_name,
            share=True,
            show_error=not opt.not_show_error,
            debug=True,
        )


