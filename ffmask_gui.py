import gradio as gr
import onnxruntime as ort
#from rembg import new_session, remove
import numpy as np
import cv2
from PIL import Image

import func_onnx, func_face, func_misc

MODELS_LIST=["u2net", "u2netp", "u2neths", "dis", "face"]
MSG_FILTER_NO_MASK="Cannot filter mask. No mask created or selected."
MSG_NO_FILTER="No filter selected/enabled."
MSG_NO_INPUT_FOR_MASK="No input image to create mask from."

# Get available ONNX execution providers
onnx_eps = ['<default>'] + [prov for prov in ort.get_available_providers()]

css = "#input-image  { width: 60% ; height: 80% ; }"

def select_mask(evt: gr.SelectData, view_mode, input_image, generated_masks):
    #print("select_mask called with event:", evt,evt.index,evt.value,evt.selected)
    idx = evt.index
    cview=update_combined_view(view_mode, input_image, generated_masks, idx)
    return cview, idx

def return_selection_index(evt: gr.SelectData):
    return evt.index

def create_mask_from_input(generated_masks,input_image, onnx_provider, model_selection):
    if input_image is None: #if there's no input image yet
        gr.Info(MSG_NO_INPUT_FOR_MASK)
        return gr.skip(), gr.skip()

    input_image=func_misc.ConvertTo_RGB_or_L(input_image)

    if model_selection == "face": #special case for face finding
        mask=func_face.GetFaceMasks_PIL(input_image)
        mask=np.array(mask,dtype=np.uint8)
    else:
        if onnx_provider == "<default>": onnx_provider = ''
        mask=func_onnx.GetMask_PIL(model_selection, onnx_provider, input_image)

    print("mask generated, shape:", mask.shape)
    if generated_masks is None: generated_masks=[(mask,None)]
    else: generated_masks.append((mask,None))
    select_idx = len(generated_masks) - 1
    mg = gr.Gallery(value=generated_masks, selected_index=select_idx)
    return mg,select_idx

def update_combined_view(view_mode, input_image, generated_masks, selected_idx):
    if view_mode == "Input": return input_image

    if generated_masks is None or len(generated_masks)==0:
        return None

    mask=generated_masks[selected_idx][0]
    print("update_combined_view mask shapes:", mask.shape,mask[:,:,0].shape)
    mask=mask[:,:,0] # Gallery converts grayscale to RGB, so we take just one channel
    if view_mode == "Mask":
        return mask

    orig_np = np.array(input_image)
    mask_np = np.array(mask) / 255.0
    masked_np = (orig_np * mask_np[:, :, np.newaxis]).astype(np.uint8)
    view_img = Image.fromarray(masked_np)
    return view_img

def update_combined_view_on_mask_selection(view_mode, input_image, generated_masks, selected_idx):
    if view_mode == "Input":
        return gr.skip()
    return update_combined_view(view_mode, input_image, generated_masks, selected_idx)

def filter_mask(generated_masks, selected_idx, dilate_image, kernel_size, iterations, invert_mask, blur_mask, blur_kernel_size):
    if generated_masks is None or len(generated_masks) == 0:
        gr.Info(MSG_FILTER_NO_MASK)
        return gr.skip(), gr.skip()
    if dilate_image==False and invert_mask==False and blur_mask==False:
        gr.Info(MSG_NO_FILTER)
        return gr.skip(), gr.skip()

    mask = generated_masks[selected_idx][0]
    if dilate_image:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    if blur_mask:
        mask=cv2.blur(mask, (blur_kernel_size, blur_kernel_size))
    if invert_mask:
        mask = 255 - mask

    generated_masks.append((mask,None)) #add to Gallery
    select_idx = len(generated_masks) - 1
    mg = gr.Gallery(value=generated_masks, selected_index=select_idx)
    return mg, select_idx

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Foreground and Face Mask Generator GUI")

    selected_idx = gr.State(value=0) # gradio doesn't let us access properties (WHY !), so we have to use a State

    with gr.Row():
        with gr.Column(scale=3):
            input_image = gr.Image(label="Input Image", sources=["upload", "clipboard"],type="pil",height=512)
            mask_gallery = gr.Gallery(value=[],label="Generated Masks", show_label=True, allow_preview=True,preview=True,
                                selected_index=None, interactive=False,height=200, columns=5, object_fit="contain",type='numpy')

        with gr.Column(scale=2):
            with gr.Row():
                onnx_provider = gr.Dropdown(choices=onnx_eps, value="<default>", label="ONNX Execution Provider",
                                        interactive=True)
                model_selection = gr.Dropdown(choices=MODELS_LIST, value="u2netp",
                                            label="Model Selection", interactive=True)
            btn_CreateMask = gr.Button("Create Mask")

            #gr.HTML("<hr>")
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

    with gr.Group():
        view_mode = gr.Radio(choices=["Masked Input", "Mask", "Input"], value="Masked Input",
                             label="View Mode", interactive=True)
        combined_view = gr.Image(label="Combined View", interactive=False,height=600)

    input_image.change( # first clear masks generatedso far
        fn=lambda: ([],0),
        inputs=[],
        outputs=[mask_gallery, selected_idx]
    ).then(
        fn=update_combined_view,
        inputs=[view_mode, input_image, mask_gallery,selected_idx],
        outputs=combined_view
    #).then(lambda: print("input_image.change() called"), inputs=[], outputs=[])
    )

    btn_ClearAll.click(
        fn=lambda: ([],0,None,None),
        inputs=[],
        outputs=[mask_gallery, selected_idx,combined_view,input_image]
    )

    btn_ClearMasks.click(
        fn=lambda: ([],0),
        inputs=[],
        outputs=[mask_gallery, selected_idx]
    ).then(
        fn=update_combined_view,
        inputs=[view_mode, input_image, mask_gallery,selected_idx],
        outputs=combined_view
    )

    btn_FilterMask.click(
        fn=filter_mask,
        inputs=[mask_gallery, selected_idx, chk_dilate, kernel_size, iterations, chk_invert, chk_blur, blur_kernel_size],
        outputs=[mask_gallery,selected_idx]
    ).then(
        fn=update_combined_view_on_mask_selection,
        inputs=[view_mode, input_image, mask_gallery,selected_idx],
        outputs=[combined_view]
    )

    btn_CreateMask.click(
        fn=create_mask_from_input,
        inputs=[mask_gallery,input_image, onnx_provider, model_selection],
        outputs=[mask_gallery,selected_idx]
    ).then(
        fn=update_combined_view,
        inputs=[view_mode, input_image, mask_gallery,selected_idx],
        outputs=combined_view
    )

    mask_gallery.select(
        fn=return_selection_index, inputs=[], outputs=selected_idx
    ).then(
        fn=update_combined_view_on_mask_selection,
        inputs=[view_mode, input_image, mask_gallery,selected_idx],
        outputs=[combined_view]
    )

    view_mode.change(fn=update_combined_view,
                     inputs=[view_mode, input_image, mask_gallery,selected_idx],
                     outputs=combined_view
    )

demo.launch()
