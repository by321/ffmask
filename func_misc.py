import os, sys
from PIL import Image
from dataclasses import dataclass

_MODELS_DIR = os.path.join(os.path.dirname(__file__),"pretrained_models")

@dataclass
class ModelInfo:
    name: str
    desc: str
    filename: str
    url: str
    width: int
    height: int
    mean: list[float]
    std: list[float]

MODELS_LIST=[
    ModelInfo('u2net' , 'U²-Net: Salient Object Detector',
              os.path.join(_MODELS_DIR,'u2net.onnx'), 'https://huggingface.co/by321/ffmask/resolve/main/u2net.onnx',
              320, 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) ),
    ModelInfo('u2netp' , 'U²-Net-p: tiny and fast version of U²-Net',
              os.path.join(_MODELS_DIR,'u2netp.onnx'), 'https://huggingface.co/by321/ffmask/resolve/main/u2netp.onnx',
              320, 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) ),
    ModelInfo('u2neths' , 'U²-Net-hs: U²-Net variant for human detection',
              os.path.join(_MODELS_DIR,'u2net_human_seg.onnx'), 'https://huggingface.co/by321/ffmask/resolve/main/u2net_human_seg.onnx',
              320, 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) ),
    ModelInfo('dis' , 'DIS: Dichotomous Image Segmentation',
              os.path.join(_MODELS_DIR,'isnet-general-use.onnx'), 'https://huggingface.co/by321/ffmask/resolve/main/isnet-general-use.onnx',
              1024, 1024, (0.5, 0.5, 0.5), (1, 1, 1) ),
    ModelInfo('birefnetlite', 'BiRefNet General, lite version',
              os.path.join(_MODELS_DIR,'BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx'),
              'https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx',
              1024, 1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) )
]

def GetModelInfo(model_name:str) -> ModelInfo:
    for m in MODELS_LIST:
        if m.name==model_name: return m
    sys.exit(f"unknown model name: {model_name}")

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

def download_missing_file(url, file_path):
    """
    If a file does not exist, download it from a URL. Shows a progress bar during download.

    Args:
        url (str): URL of the file to download
        file_path (str): Local path where the file should be saved

    Exits program on any error with exception message.
    """
    import requests
    from tqdm import tqdm
    try:
        if os.path.exists(file_path): return

        print(f"downloading from: {url}")
        print(f"  local filename: {file_path}")
		# Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an error for bad status codes

        total_size = int(response.headers.get('content-length', 0))

        # Download file in chunks, set chunk size to 1/16 of total size, capped at 1 MB
        chunk_size = min(total_size // 16, 1048576) if total_size > 0 else 1048576

        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(file_path))

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

    except Exception as e:
        print(f"Download error: {e}")
        sys.exit(1)
