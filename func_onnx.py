import sys, cv2, tqdm
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

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

def preprocess_grayscale_image(image,isPilImage=False):
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
    mean = 0.485
    std = 0.229
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=2)  # Shape: [height, width, 1]
    img_array = np.repeat(img_array, 3, axis=2)    # Shape: [height, width, 3]
    img_array = img_array.transpose(2, 0, 1)  # Shape: [3, height, width]
    img_array = np.expand_dims(img_array, axis=0)  # Shape: [1, 3, height, width]
    return img_array

def preprocess_rgb_image(image):
    """
    Process an image with resizing, per-channel max and mean-std normalization.

    Args:
        image: PIL.Image or numpy array image, in RGB mode

    Returns:
        numpy.ndarray, shape [1, 3, height, width], processed image
    """

    img_array = np.array(image, dtype=np.float32)  # Shape: [height, width, 3]
    max_vals = img_array.max(axis=(0, 1))  # Shape: [3]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    for c in range(3):
        if max_vals[c] > 0:
            img_array[:, :, c] = (img_array[:, :, c] / max_vals[c] - mean[:, :, c]) / std[:, :, c]
    img_array = img_array.transpose(2, 0, 1)  # Shape: [3, height, width]
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

def ProcessOneVideo(input_file, output_file, model, ep):
    """Process an input video and save to output video in MP4 format."""
    import func_misc
    onnx_model_path=func_misc.GetDefaultU2NetModelPath_ONNX(model,ensure_exists=True)
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
    update_interval=10

    # Define the codec (H.264 for high quality, fallback to mp4v if needed)
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)
    if not out.isOpened():
        print(f"could not create output video file '{output_file}'",file=sys.stderr)
        cap.release()
        sys.exit(-2)

    print(f"processing video '{input_file}' with model '{model}'...")
    if exec_provider=='':
        ort_session = ort.InferenceSession(onnx_model_path)
    else:
        ort_session = ort.InferenceSession(onnx_model_path,providers=[exec_provider])
    print(ort_session.get_inputs()[0])
    pbar= tqdm.tqdm(total=total_frames, desc="Processing frames", unit="frame")

    # Process frames
    frame_count = 0; isGrayScale=False
    while True:
        ret, frame = cap.read()
        if not ret: break
        print_frame_info(frame,"cap.read")

        if 0==frame_count:
            isGrayScale=IsFrameGrayScale(frame)
            #print(f"\n\ngrayscale flag: {isGrayScale}\n\n")
        h, w, n = frame.shape#; print("w h extracted:",h,w,n)
        img320=cv2.resize(frame, (320,320), interpolation=cv2.INTER_LINEAR)
        print_frame_info(img320,"img320")
        if isGrayScale:
            x=preprocess_grayscale_image(img320)
        else:
            x=preprocess_rgb_image(cv2.cvtColor(img320, cv2.COLOR_BGR2RGB))
        print_frame_info(x,"x")

        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outs = ort_session.run(None, ort_inputs)
        d1 = ort_outs[0].squeeze()
        print_frame_info(d1,"ort_outs[0].squeeze()")
        ma=np.max(d1)
        mi=np.min(d1)
        #print(f"ma {ma} mi {mi}")
        if (ma!=mi):
            processed_frame=255.0*(d1-mi)/(ma-mi) #normalize
            processed_frame = processed_frame.astype(np.uint8)
            processed_frame = cv2.resize(processed_frame, (w,h), interpolation=cv2.INTER_CUBIC)
            print_frame_info(processed_frame,"processed_frame")
        else:
            processed_frame=np.zeros((h,w), dtype=np.uint8)

        out.write(processed_frame)
        frame_count += 1
        # Update progress bar every N frames
        if frame_count % update_interval == 0 or frame_count == total_frames:
            pbar.update( frame_count - pbar.n)

    pbar.close()
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames. Done.")

def ProcessOneImage(inimage, outimage, model, ep, alpha_png):
    import func_misc, time
    onnx_model_path=func_misc.GetDefaultU2NetModelPath_ONNX(model,ensure_exists=True)
    exec_provider=get_execution_provider_by_partial_name(ep)

    img=func_misc.LoadInputImage(inimage)
    img320 = img.resize((320, 320), Image.Resampling.LANCZOS)
    #print(f"resized image size: {img320.size}, mode: {img320.mode}")
    if img320.mode == 'L':
        x=preprocess_grayscale_image(img320,isPilImage=True)
    else:
        x=preprocess_rgb_image(img320)
    #print(f"input image size: {x.shape}, dtype: {x.dtype}")
    if exec_provider=='':
        ort_session = ort.InferenceSession(onnx_model_path)
    else:
        ort_session = ort.InferenceSession(onnx_model_path,providers=[exec_provider])
    #print(ort_session.get_inputs()[0])
    t1=time.perf_counter()
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    d1 = ort_outs[0].squeeze()
    t1=time.perf_counter()-t1
    print(f"model inference time: {t1*1000:.1f} ms")
    #print(type(d1),d1.shape )
    ma=np.max(d1)
    mi=np.min(d1)
    #print(f"ma {ma} mi {mi}")
    if (ma!=mi):
        d1=255.0*(d1-mi)/(ma-mi)
        im=Image.fromarray(d1.astype(np.uint8), mode='L')
    else:
        im=Image.new(mode='L', size=d1.shape, color=0)
    #if theCtx['invert_mask']: im=ImageOps.invert(im)
    del d1

    maskImg=im.resize(img.size,resample=Image.LANCZOS)
    func_misc.SaveOutputImage(img, maskImg, outimage, alpha_png)
