import os,sys, time, cv2, tqdm
import numpy as np
from PIL import Image
#from PIL import Image, ImageOps
import onnxruntime as ort
import func_misc


def CreateOnnxSession(exec_provider,theModel):
    print(f"starting ONNX session, execution provider='{exec_provider}' model file='{theModel.filename}'")
    if not os.path.exists(theModel.filename):
        print("model file does not exist, starting download...")
        func_misc.download_missing_file(theModel.url, theModel.filename)

    if exec_provider=='':
        ort_session = ort.InferenceSession(theModel.filename)
    else:
        ort_session = ort.InferenceSession(theModel.filename,providers=[exec_provider])
    print("model input:",ort_session.get_inputs()[0])
    return ort_session

def RunOnnxSession(ort_session:ort.InferenceSession, input_data, printInfo=False):
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    if printInfo:print("*** ort_session.run() returned ort_outs[0].shape:",ort_outs[0].shape)
    d1 = ort_outs[0][:,0,:,:].squeeze()
    return d1

def PostProcessMask(model_name,mask,imgw,imgh):
    if model_name.startswith("birefnet"): mask=sigmoid(mask)
    ma=np.max(mask)
    mi=np.min(mask)
    if (ma!=mi):
        mask=255.0*(mask-mi)/(ma-mi)
        mask = cv2.resize(mask.astype(np.uint8), (imgw,imgh), interpolation=cv2.INTER_CUBIC)
    else:
        mask= np.zeros((imgh, imgw), dtype=np.uint8) #numpy size is height first
    return mask

def get_execution_provider_by_partial_name(ep):
    available_providers = ort.get_available_providers()
    if len(available_providers)==0:
        sys.exit("no ONNX execution providers available")

    if ep=='':
        print("no ONNX execution provider specified, using default (probably CPU)")
        return ''

    n2=ep.lower()
    for provider in available_providers:
        if n2 in provider.lower():
            print(f"using execution provider '{provider}'")
            return provider
    sys.exit(f"could not find ONNX execution provider name matching '{ep}'\n"
             f"available providers are: {available_providers}")

def preprocess_grayscale_image(image, pre_mean:list[float], pre_std:list[float], isPilImage=False):
    """
    Process a grayscale PIL image with resizing, max normalization,
    mean-std normalization, and channel expansion.

    Args:
        image: PIL.Image or 2D numpy array image, grayscale (mode 'L')

    Returns:
        numpy.ndarray, shape [1, 3, height, width], processed image
    """
    if isPilImage:
        max_val = image.getextrema()[1]  # getextrema() returns (min, max) for grayscale
    else:
        max_val = np.max(image)
    if max_val == 0:
        w, h = image.size
        return np.zeros((1, 3, h, w), dtype=np.float32)

    img_array = np.array(image, dtype=np.float32)
    img_array /= max_val

    # Apply mean-std normalization
    img_array = (img_array - pre_mean[0]) / pre_std[0]
    #print("shape before",img_array.shape)
    img_array = np.repeat(img_array[np.newaxis, :, :], 3, axis=0)  # irectly to CHW
    #print("shape after",img_array.shape)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: [1, 3, height, width]
    return img_array

def preprocess_rgb_image(image, pre_mean:list[float], pre_std:list[float]):
    """
    Process an image with resizing, per-channel max and mean-std normalization.

    Args:
        image: PIL.Image or numpy array image, in RGB mode

    Returns:
        numpy.ndarray, shape [1, 3, height, width], processed image
    """
    img_array = np.array(image, dtype=np.float32)  # Shape: [height, width, 3]
    max_vals = img_array.max(axis=(0, 1))  # Shape: [3]
    mean = np.array(pre_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(pre_std, dtype=np.float32).reshape(1, 1, 3)
    for c in range(3):
        if max_vals[c] > 0:
            img_array[:, :, c] = (img_array[:, :, c] / max_vals[c] - mean[:, :, c]) / std[:, :, c]
    img_array = img_array.transpose(2, 0, 1)  # Shape: [3, height, width], HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Shape: [1, 3, height, width]
    return img_array

def IsFrameGrayScale(frame):
    if frame.dtype != np.uint8:
        sys.exit(f"cv2 returned unexpected video frame data type: {frame.dtype}")
    if len(frame.shape) == 2: return True
    if len(frame.shape) == 3:
        if frame.shape[2] == 3:
            return False

    sys.exit(f"cv2 returned unexpected video frame shape: {frame.shape}")

def print_frame_info(f,name):
    #print(f"name:{name}, shape: {f.shape}, dtype: {f.dtype}")
    pass
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ProcessOneVideo(input_file, output_file, model, ep):
    """Process an input video and save to output video in MP4 format."""
    theModel=func_misc.GetModelInfo(model)
    exec_provider=get_execution_provider_by_partial_name(ep)

    # Open the input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"could not open input video file '{input_file}'",file=sys.stderr)
        sys.exit(-1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count

    # Define the codec (H.264 for high quality, fallback to mp4v if needed)
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)
    if not out.isOpened():
        print(f"could not create output video file '{output_file}'",file=sys.stderr)
        cap.release()
        sys.exit(-2)

    ort_session=CreateOnnxSession(exec_provider,theModel)

    pbar= tqdm.tqdm(total=total_frames, desc="Processing frames", unit="frame")
    # Process frames
    frame_count = 0; isGrayScale=False
    pre_mean=theModel.mean
    pre_std=theModel.std
    model_input_size=(theModel.width,theModel.height)
    update_interval=2
    last_update_time = time.time()-update_interval-1
    while True:
        ret, frame = cap.read()
        if not ret: break
        print_frame_info(frame,"cap.read")

        if 0==frame_count:
            isGrayScale=IsFrameGrayScale(frame)
            #print(f"\n\ngrayscale flag: {isGrayScale}\n\n")
        h, w, n = frame.shape#; print("w h extracted:",h,w,n)
        img_resized=cv2.resize(frame, model_input_size, interpolation=cv2.INTER_LINEAR)
        print_frame_info(img_resized,"img_resized")
        if isGrayScale:
            x=preprocess_grayscale_image(img_resized,pre_mean,pre_std)
        else:
            x=preprocess_rgb_image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB),pre_mean,pre_std)
        print_frame_info(x,"x")

        d1=RunOnnxSession(ort_session,x)
        processed_frame=PostProcessMask(theModel.name,d1,w,h)

        out.write(processed_frame)
        frame_count += 1

        tn=time.time()
        if tn-last_update_time > update_interval:
            last_update_time = tn
            if frame_count<=total_frames: #reported total frames can be wrong
                pbar.update( frame_count - pbar.n)
            else:
                pbar.update( total_frames - pbar.n)

    pbar.close()
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames. Done.")

def GetMask_PIL(model, exec_provider, img):
    theModel=func_misc.GetModelInfo(model)

    model_input_size=(theModel.width,theModel.height)
    img_resized = img.resize(model_input_size, Image.Resampling.LANCZOS)
    #print(f"resized image size: {img_resized.size}, mode: {img_resized.mode}")
    if img_resized.mode == 'L':
        x=preprocess_grayscale_image(img_resized,theModel.mean,theModel.std,isPilImage=True)
    else:
        x=preprocess_rgb_image(img_resized,theModel.mean,theModel.std)

    ort_session=CreateOnnxSession(exec_provider,theModel)

    #print(ort_session.get_inputs()[0])
    t1=time.perf_counter()
    d1=RunOnnxSession(ort_session,x,printInfo=True)
    t1=time.perf_counter()-t1
    print(f"model inference time: {t1*1000:.1f} ms")
    return PostProcessMask(model,d1,img.width,img.height)

def ProcessOneImage(inimage, outimage, model, ep, alpha_png):
    exec_provider=get_execution_provider_by_partial_name(ep)
    img=func_misc.LoadInputImage(inimage)
    npimg=GetMask_PIL(model, exec_provider, img)
    maskImg=Image.fromarray(npimg, mode='L')
    func_misc.SaveOutputImage(img, maskImg, outimage, alpha_png)

