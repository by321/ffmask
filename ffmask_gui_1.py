import subprocess, os, platform, cv2, time
from PIL import Image
import gradio as gr
import onnxruntime as ort
import numpy as np

import func_onnx, func_face, func_misc


#ONNX execution providers
ONNX_EPS = ['<default>'] + [prov for prov in ort.get_available_providers()]

MODELS=[(x.name,x.desc) for x in func_misc.MODELS_LIST]
MODELS.append(("face","Face Mask"))
MODELS_DESC=(m[1] for m in MODELS)
MSG_FILTER_NO_MASK="Cannot filter mask. No mask created or selected."
MSG_NO_FILTER="No filter selected/enabled."
MSG_NO_INPUT_FOR_MASK="No input image to create mask from."

VIEW_MODE_MIBC="Masked Input over BG Color"
VIEW_MODE_MIBI="Masked Input over BG Image"

#Unlike what the documentation says, gradio's ColorPicker actually returns 4 kinds of strings:
# color changed to: 'rgba(49.836245031524115, 105.8758538593013, 134.61874999999998, 1)' type=<class 'str'>
# color changed to: '#326b8' type=<class 'str'>
# color changed to: 'rgb(102, 0, 71)' type=<class 'str'>
# color changed to: 'hsl(0, 10%, 20%)' type=<class 'str'>
#This function handles 3 of them, but not the HSL format
def parse_color_str(clrstr):
    if clrstr[0]=='#': return clrstr #looks good already
    if not clrstr.startswith('rgb'): # neither rgb(r,g,b) nor rgba(r,g,b,a)
        gr.Warning(f"Unsupported color format: {clrstr}")
        return "#000000"

    # extract substring inside ( ), then split by comma
    values=clrstr.split('(')[1].split(')')[0].split(',')

    r = min(max(round(float(values[0].strip())), 0), 255)
    g = min(max(round(float(values[1].strip())), 0), 255)
    b = min(max(round(float(values[2].strip())), 0), 255)
    return f"#{r:02x}{g:02x}{b:02x}" # to #rrggbb format

def return_selection_index(evt: gr.SelectData):
    print("return_selection_index() called, index:", evt.index)
    return evt.index

def create_mask_from_input(generated_masks,input_image, onnx_provider, model_selection):
    print("create_mask_from_input() called")
    if input_image is None: #if there's no input image yet
        gr.Info(MSG_NO_INPUT_FOR_MASK)
        return gr.skip(), gr.skip()

    input_image=func_misc.ConvertTo_RGB_or_L(input_image) #input should be RGB, but just in case
    model_name=MODELS[model_selection][0]
    print("creating mask using model index:", model_selection,'name:',model_name)
    if model_name == "face": #special case for face finding
        mask=func_face.GetFaceMasks_PIL(input_image)
        mask=np.array(mask,dtype=np.uint8)
    else:
        if onnx_provider == "<default>": onnx_provider = ''
        mask=func_onnx.GetMask_PIL(model_name, onnx_provider, input_image)

    print("mask generated, shape:", mask.shape)
    img=Image.fromarray(mask)
    if generated_masks is None: generated_masks=[(img,None)]
    else: generated_masks.append((img,None))
    select_idx = len(generated_masks) - 1
    mg = gr.Gallery(value=generated_masks, selected_index=select_idx)
    return mg,select_idx

def update_combined_view(view_mode, input_image, generated_masks, selected_idx, bg_color,bg_image):
    print(f"update_combined_view(), view_mode={view_mode}")

    if view_mode == "Input": return input_image

    #the remaining cases all need a mask selected
    if generated_masks is None or len(generated_masks)==0:
        print("  no masks generated yet, returning")
        return None

    print("update_combined_view input image:", input_image.size,input_image.mode)
    mask=generated_masks[selected_idx][0]
    if view_mode == "Mask": return mask

    if mask.mode!='L': # Gallery converts grayscale to RGB, so we take just one channel
        mask=mask.getchannel(0)

    if view_mode == VIEW_MODE_MIBC: #view mode is masked input over background color
        clrstr= parse_color_str(bg_color)
        inew=Image.new(input_image.mode, input_image.size, clrstr)
        t1=time.perf_counter()
        inew.paste(input_image,None,mask)
        t1=time.perf_counter()-t1
    else: #view mode is masked input over background image
        if bg_image is None: return None #no background image yet
        inew=bg_image
        if inew.size != input_image.size:
            inew=inew.resize(input_image.size, Image.LANCZOS)
        t1=time.perf_counter()
        inew.paste(input_image,None,mask)
        t1=time.perf_counter()-t1

    print(f"image merge time: {t1*1000:.1f} ms")
    return inew
    '''
    orig_np = np.array(input_image)
    t1=time.perf_counter()
    masked_np = ((orig_np.astype(np.uint16) * mask[:, :, np.newaxis]+127) / 255).astype(np.uint8)
    #mask_np = np.array(mask) / 255.0
    #masked_np = (orig_np * mask_np[:, :, np.newaxis]).astype(np.uint8)
    t1=time.perf_counter()-t1
    print(f"image merge time: {t1*1000:.1f} ms")
    view_img = Image.fromarray(masked_np)
    return view_img
    '''

def filter_mask(generated_masks, selected_idx, dilate_image, kernel_size, iterations, invert_mask, blur_mask, blur_kernel_size):
    if generated_masks is None or len(generated_masks) == 0:
        gr.Info(MSG_FILTER_NO_MASK)
        return gr.skip(), gr.skip()
    if dilate_image==False and invert_mask==False and blur_mask==False:
        gr.Info(MSG_NO_FILTER)
        return gr.skip(), gr.skip()

    mask = generated_masks[selected_idx][0]
    if mask.mode != 'L':  # Gallery converts grayscale to RGB, so we take just one channel
        mask = mask.getchannel(0)
    mask = np.array(mask, dtype=np.uint8)
    if dilate_image:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    if blur_mask:
        mask=cv2.blur(mask, (blur_kernel_size, blur_kernel_size))
    if invert_mask:
        mask = 255 - mask
    mask=Image.fromarray(mask)
    generated_masks.append((mask,None)) #add to Gallery
    select_idx = len(generated_masks) - 1
    mg = gr.Gallery(value=generated_masks, selected_index=select_idx)
    return mg, select_idx

with gr.Blocks(fill_width=True) as demo:
    selected_idx = gr.State(value=0) # gradio doesn't let us access properties (WHY !), so we have to use a State
    bg_image=gr.State(value=None) #background image for combined view
    gr.Markdown("## Foreground and Face Mask Generator GUI")
    with gr.Row():
        with gr.Column(scale=3):
            input_image = gr.Image(label="Input Image", sources=["upload", "clipboard"],type="pil",height=512,image_mode="RGB")
            mask_gallery = gr.Gallery(value=[],label="Generated Masks", show_label=True, allow_preview=True,preview=True,
                                selected_index=None, interactive=False,height=200, columns=5, object_fit="contain",type='pil')

        with gr.Column(scale=2):
            with gr.Row():
                onnx_provider = gr.Dropdown(choices=ONNX_EPS, value="<default>", label="ONNX Provider",
                                            interactive=True,scale=2)
                model_selection = gr.Dropdown(choices=MODELS_DESC, value=MODELS[0][1], type="index",
                                            label="Model Selection", interactive=True,scale=3)
            with gr.Row():
                btn_CreateMask = gr.Button("Create Mask",scale=3)
                gr.Button("Models Info").click( fn=None,
                    js="() => window.open('https://html-preview.github.io/?url=https://github.com/by321/ffmask/blob/main/static/models_info.html', '_blank')" )

            gr.Markdown("### Filter Selected Mask:",container=False)
            with gr.Group():
                chk_dilate = gr.Checkbox(label="Dilate Mask", value=False, container=False)
                with gr.Row():
                    kernel_size = gr.Slider(minimum=3, maximum=9, step=2, value=3, label="Kernel Size",interactive=True)
                    iterations = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Iterations",interactive=True)
            with gr.Group():
                chk_blur=gr.Checkbox(label="Blur Mask", value=False, container=False)
                blur_kernel_size=gr.Slider(minimum=3, maximum=15, step=2, value=3, label="Kernel Size",interactive=True)
            chk_invert = gr.Checkbox(label="Invert Mask", value=False, container=False)
            btn_FilterMask=gr.Button("Filter Mask" )

            #gr.HTML("<hr>")
            with gr.Row():
                btn_ClearMasks = gr.Button("Clear Masks")
                btn_ClearAll = gr.Button("Clear All")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                view_mode = gr.Radio(choices=[VIEW_MODE_MIBC,VIEW_MODE_MIBI, "Mask", "Input"], value=VIEW_MODE_MIBC,
                                    label="View Mode", interactive=True)
                combined_view = gr.Image(label="Combined View", interactive=False,height=600,type='pil')
        with gr.Column(scale=2):
            with gr.Group():
                bg_color = gr.ColorPicker(label="Background Color", value="#000000", interactive=True)
                bg_image = gr.Image(label="Background Image", sources=["upload", "clipboard"],height=600,type="pil",image_mode="RGB")

    #inputs to update_combined_view()
    update_combined_view_inputs=[ view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image]

    btn_CreateMask.click( # create mask button clicked
        fn=create_mask_from_input, #this will trigger mask_gallery.select
        inputs=[mask_gallery,input_image, onnx_provider, model_selection],
        outputs=[mask_gallery,selected_idx]
    )

    input_image.input( # input image changed by user
        fn=lambda: ([],0), # first clear masks generatedso far
        inputs=[],
        outputs=[mask_gallery, selected_idx]
    ).then(
        fn=update_combined_view,
        #inputs=[ view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image],
        inputs=update_combined_view_inputs,
        outputs=[combined_view]
    #).then(lambda: print("input_image.change() called"), inputs=[], outputs=[])
    )

    def _bg_color_input_update_view(view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image):
        if view_mode != VIEW_MODE_MIBC: return gr.skip() #not masked input over background color
        return update_combined_view(view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image)
    bg_color.input( #background color changed by user
        fn=_bg_color_input_update_view,
        inputs=update_combined_view_inputs,
        outputs=[combined_view]
    )

    def _bg_image_input_update_view(view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image):
        if view_mode != VIEW_MODE_MIBI: return gr.skip() #not masked input over background image
        return update_combined_view(view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image)
    bg_image.input( #background image changed by user
        fn=_bg_image_input_update_view,
        inputs=update_combined_view_inputs,
        outputs=[combined_view]
    )

    btn_ClearAll.click( #clear all button clicked
        fn=lambda: ([],0,None,None, None),
        inputs=[],
        outputs=[mask_gallery, selected_idx,combined_view,input_image, bg_image]
    )

    btn_ClearMasks.click( #clear mask button clicked
        fn=lambda: ([],0),
        inputs=[],
        outputs=[mask_gallery, selected_idx]
    ).then(
        fn=update_combined_view,
        inputs=update_combined_view_inputs,
        outputs=[combined_view]
    )

    btn_FilterMask.click( #filter mask button clicked
        fn=filter_mask, #this will trigger mask_gallery.select
        inputs=[mask_gallery, selected_idx, chk_dilate, kernel_size, iterations, chk_invert, chk_blur, blur_kernel_size],
        outputs=[mask_gallery,selected_idx]
    )

    def _mask_gallery_select_update_view(view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image):
        if view_mode == "Input": return gr.skip() #no need to update combined view
        return update_combined_view(view_mode, input_image, mask_gallery,selected_idx, bg_color,bg_image)
    mask_gallery.select( #new mask selected
        fn=return_selection_index, inputs=[], outputs=selected_idx
    ).then(
        fn=_mask_gallery_select_update_view,
        inputs=update_combined_view_inputs,
        outputs=[combined_view]
    )

    view_mode.input( #combined view mode changed
        fn=update_combined_view,
        inputs=update_combined_view_inputs,
        outputs=[combined_view]
    )
demo.launch()
