# from share import *
import config

import argparse
import matplotlib
import cv2
import einops
import re
import gradio as gr
import numpy as np
import torch
from PIL import Image

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from functools import partial
from deepfloyd_if.modules.stage_II import IFStageII

grad_channels = 0

stage_2 = 0

MAX_COLORS = 128


def load(model_load_path, sam_path):

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




def sort_colors(color_array):
        # Ensure the color values are in the range [0, 255] and the correct data type
        color_array = np.clip(color_array, 0, 255).astype(np.uint8)
        
        # Reshape the array to a 2D array with a single row
        reshaped_array = color_array.reshape((-1, 1, 3))
        
        # Convert RGB to HSV
        hsv_colors = cv2.cvtColor(reshaped_array, cv2.COLOR_RGB2HSV)
        
        # Reshape back to a 2D array
        hsv_colors = hsv_colors.reshape(-1, 3)
        
        # Sort primarily by hue, then by saturation, then by value
        sorted_indices = np.lexsort((hsv_colors[:, 2], hsv_colors[:, 1], hsv_colors[:, 0]))
        
        # Apply the sorting to the original RGB array
        sorted_colors = color_array[sorted_indices]
        
        return sorted_colors


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


def apply_palette(pil_image, pil_target, num_colors, transfer_method, cmap, blend): #image and Pal are HWC 0-255 RGB images        
        height, width = pil_image.size

        if num_colors == None:
             raise ValueError("Invalid value provided for colors")
             
        num_colors = int(num_colors)
        palette_image = pil_image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        pal_rgb = palette_image.convert('RGB')
        pal_rgb = np.asarray(pal_rgb)
        indexed_palette = np.array(palette_image)
        dst_pal, dst_rgb = 0, 0
        quantized_source_image, augmented_image_rgb, quantized_augmented_image, source_image = 0, 0, 0, 0

        srcPalette3 = np.array(palette_image.getpalette()[0:3*num_colors]).reshape(num_colors,3)/255.
        # Get the dimensions of pal_rgb
        height, width = indexed_palette.shape[:2]
        fullColor = np.zeros((height,width,3),dtype=np.float32)
        if transfer_method == 'colormap': #for color map, we're doing most similar color
            mapQuery = np.arange(0,1 + (1/(2.*num_colors)),1./(num_colors-1))
            cmap = matplotlib.colormaps[cmap]
            cm = cmap(mapQuery)[:,0:3]
            D = np.exp(-distance_matrix(srcPalette3, cm, p=2))

            # Create the discrete palette
            discrete_indices = np.repeat(np.arange(num_colors), height // num_colors)
            if len(discrete_indices) < height:
                discrete_indices = np.pad(discrete_indices, (0, height - len(discrete_indices)), mode='edge')
            discrete_palette = cm[discrete_indices] 
            dst_pal = np.repeat(discrete_palette[:, np.newaxis, :], width, axis=1)

            # Create the smoothed color map
            smoothed_indices = np.linspace(0, 1, height)
            smoothed_colormap = cmap(smoothed_indices)[:, :3]
            dst_rgb = np.repeat(smoothed_colormap[:, np.newaxis, :], width, axis=1)

            # Create a gradient from 0 to 1
            gradient = np.linspace(0, 1, 256)
            gradient = np.tile(gradient, (50, 1))  # shape (50, 256)


            # Apply the colormap
            colored_gradient = cmap(gradient)  # returns RGBA values

            # Convert to 8-bit RGB
            rgb_gradient = (colored_gradient[:, :, :3] * 255).astype(np.uint8)

            # Create a PIL Image
            targ_rgb = Image.fromarray(rgb_gradient)


        else:
            targInt = pil_target.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
            targ_rgb = np.array(targInt.convert('RGB'))
            targPalette3 = np.array(targInt.getpalette()[0:3*num_colors]).reshape(num_colors,3)/255.
            targCol = targ_rgb / 255.
            targInt = np.array(targInt)

            if transfer_method == 'color':
                D = np.exp(-distance_matrix(srcPalette3, targPalette3,p=2))

            elif transfer_method == 'negative':
                D = np.exp(distance_matrix(srcPalette3, targPalette3,p=2))

            elif transfer_method == 'frequency':
                sortedSrc, countsSrc = np.unique(indexed_palette, return_counts=True)
                sortedTarg, countsTarg = np.unique(targInt, return_counts=True)
                A = np.reshape(countsSrc/float(np.sum(countsSrc)),(num_colors,1))
                B = np.reshape(countsTarg/float(np.sum(countsTarg)),(num_colors,1))
                D = np.exp(-distance_matrix(A, B,p=2))

            elif transfer_method == 'int':
                D = np.exp(-distance_matrix(np.sum(srcPalette3,1,keepdims=True), np.sum(targPalette3,1,keepdims=True),p=2))
                
            # Create an array of row indices
            row_indices = np.arange(height)
            # Calculate which color each row should be (integer division)
            color_indices = row_indices * num_colors // height
    
            # Use advanced indexing to create the image array
            targPalette3 = sort_colors(targPalette3*255) / 255.
            image_array = targPalette3[color_indices][:, np.newaxis, :]
            
            # Repeat the colors across all columns
            dst_pal = np.repeat(image_array, width, axis=1)

            dst_rgb = np.asarray(pil_target)
                        
        _, matching = linear_sum_assignment(-D)

        for colIDX in range(num_colors):
            mask = indexed_palette == colIDX
            if transfer_method == 'colormap':
                getColor = cm[matching[colIDX],:]
            else:
                getMask = targInt==matching[colIDX]
                getColor = targCol[getMask][0] #get one instance of this color
            fullColor[mask] = blend * getColor + (1-blend) * srcPalette3[colIDX]

        return [pal_rgb, targ_rgb, (fullColor*255).astype(np.uint8)] #palettized src image, palettized style, style image unmodified
def getGradResize(input_image, image_resolution, mode):
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

    return resizedIm


def process(palette, num_colors, textureOption):
    with torch.no_grad():
        effective_batch_size = 1
        steps = "super27"
        aug_level =  0.0
        support_noise_less_qsample_steps = 0
        dynamic_thresholding_p = 0.95
        dynamic_thresholding_c =  1.0
        sample_loop = 'ddpm'
        num_colors = int(num_colors)
        pal = np.asarray(palette)

        height, width = pal.shape[:2]
        
        detected_map = np.ones((height, width, 7), dtype=np.float32) #7 channels in the conditioning
        

        texture_channel = np.ones((height, width))
        if textureOption:
            texture_channel[:,:] = 0

        color_indicator = np.full((height, width), num_colors / (2 * MAX_COLORS))

        detected_map = np.ones((height, width, 7))

        detected_map[:, :, :3] = np.asarray(palette)
        detected_map[:, :, 3:5] = grad_channels #(grad[0] +255)/2 #
     

        detected_map[:, :, 5] = color_indicator
        detected_map[:, :, 6] = texture_channel


    
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




opt = app_options()

img_with_mask = partial(gr.Image, type="pil", height=300, interactive=True, show_label=True)

with gr.Blocks(
        title="Colorize Diffusion",
        theme=gr.themes.Soft(),
        elem_id="main-interface",
        analytics_enabled=False
) as block:
    with gr.Row(elem_id="content-row", equal_height=False, variant="panel"):
                with gr.Column():
                    image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1088, value=256, step=64)
                    model_path = gr.Textbox(value="/home/faariss2/G.pt", label="model_path")
                    load_models = gr.Button("Load Models", variant="secondary", size="sm")
                    mode = gr.Radio(["Luminance", "Gradient", "Threshold"], label="Mode")
                    transfer_method = gr.Radio(["colormap", "color", "frequency", "negative", "int"], label="method")
                    colors = gr.Radio(["16", "32", "64"], label="Number of Colors")
                    cmap = gr.Textbox(value="viridis", label="colormap")
                    blend = gr.Slider(minimum=0.0, maximum=1, label="blend", value=1)
                    
                with gr.Column():
                    input_image = gr.Image(label = "Source", sources='upload', type="pil")
                    ColorImage =  gr.Image(label = "Target", sources='upload', type="pil")
                    quantize = gr.Button(value="Quantize")
                    q_source = gr.Image(label = "Quantized Source", sources='upload', type="pil")
                    q_target = gr.Image(label = "Quantized Target", sources='upload', type="pil")
                    palette = gr.Image(label = "Palette",sources='upload', type="pil")
                    # with gr.Accordion("Advanced options", open=False):
                        # steps = gr.Textbox(value="super27", label="steps")
                        # aug_level = gr.Number(value = 0.0, label="aug_level")
                        # support_noise_less_qsample_steps = gr.Number(value = 0, label="support_noise_less_qsample_steps")
                        # dynamic_thresholding_p = gr.Number(value = 0.95,label="dynamic_thresholding_p")
                        # dynamic_thresholding_c = gr.Number(value = 1.0,label="dynamic_thresholding_p")
                        # sample_loop = gr.Textbox(value='ddpm', label="sample_loop")

                    textureOption = gr.Checkbox(label="Texture Dropout", info="Check to Dropout Texture Layer")
                    run_button = gr.Button("🚀 Generate", variant="primary", size="lg")


                with gr.Column():
                    result = gr.Gallery(label='Generated Image', show_label=True, elem_id="gallery", preview=True)


            
  

    run_button.click(fn=process, inputs=[palette, colors, textureOption], outputs=[result])
    quantize.click(fn=apply_palette, inputs=[input_image,ColorImage, colors, transfer_method, cmap, blend], outputs=[q_source, q_target, palette])
    input_image.input(fn=getGradResize, inputs=[input_image, image_resolution, mode], outputs=[input_image])
    load_models.click(fn=load, inputs=[model_path])

    block.launch(
            server_name=opt.server_name,
            share=True,
            show_error=not opt.not_show_error,
            debug=True,
        )


