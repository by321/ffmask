import os, sys
from PIL import Image

def ConvertTo_RGB_or_L(img:Image) -> Image:
    if img.mode == "L" or img.mode == "RGB": return img

    tgtmode='RGB' # by default convert to RGB
    if img.mode=="LA" or img.mode=="La": # L with alpha
        tgtmode='L'

    print(f"  * converting from mode {img.mode} to {tgtmode}")
    return img.convert(tgtmode)

def LoadInputImage(input_file:str) -> Image:
    try:
        i1 = Image.open(input_file)
        print(f"input image: {input_file}, size: {i1.size}, mode: {i1.mode}")
        i1=ConvertTo_RGB_or_L(i1)
    except Exception as inst:
        print(type(inst),':',inst, file=sys.stderr)
        print("failed to load input image:",input_file, file=sys.stderr)
        quit()
    return i1

def SaveOutputImage(img:Image, maskImg:Image, out_fn:str, alpha_png:bool):
    if alpha_png:
        img.putalpha(maskImg)
        print(f"saving PNG file with mask as alpha channel: {out_fn}")
        img.save(out_fn,format="PNG")
    else:
        print(f"saving mask image: {out_fn}")
        maskImg.save(out_fn)

DIS_U2Net_pth_files={'dis':'isnet-general-use.pth', 'u2net':'u2net.pth', 'u2netp':'u2netp.pth', 'u2neths':'u2net_human_seg.pth'}

def GetDefaultU2NetModelPath_Torch(model_name:str):
    if model_name not in DIS_U2Net_pth_files:
        sys.exit(f"unexpected U2Net model name: {model_name}")
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir,"pretrained_models",DIS_U2Net_pth_files[model_name])

def GetDefaultU2NetModelPath_ONNX(model_name:str,ensure_exists:bool=False):
    x = GetDefaultU2NetModelPath_Torch(model_name) + ".onnx"
    if ensure_exists and not os.path.isfile(x):
        sys.exit(f"ONNX model file doesn't exist: {x}")
    return x

