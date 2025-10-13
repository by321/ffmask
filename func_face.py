import os, sys
import cv2, tqdm
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import face_recognition
import numpy as np

import func_misc

def GetHaarCascade():
    current_dir = os.path.dirname(__file__)
    full_model_path=os.path.join(current_dir,"pretrained_models","haarcascade_frontalface_alt2.xml")
    if not os.path.isfile(full_model_path):
        sys.exit(f"Haar Cascade model file doesn't exist: {full_model_path}")

    haar_cascade = cv2.CascadeClassifier(full_model_path)
    return haar_cascade

def ExpandHCFaceRect(x,y,w,h,framew,frameh):
    x0 = max(round(x-w/3)  ,0)
    x1 = min(round(x+w+w/3),framew)
    y0 = max(round(y-h/2)  ,0) # more
    y1 = min(round(y+h+h/3),frameh)
    return x0,y0,x1,y1

def pil_to_cv2(pil_image):
    if pil_image.mode not in ('L', 'RGB'):
        raise ValueError("Image must be in 'L' (grayscale) or 'RGB' mode")
    img_array = np.array(pil_image, dtype=np.uint8)
    if pil_image.mode == 'L': return img_array
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def _convex_hull_from_face_landmarks(face_landmarks, offset_x, offset_y, scale_factor=1.05):
    pts=[]
    for x in face_landmarks.values(): pts=pts+x #put all points in pts[]
    vertices=cv2.convexHull(np.asarray(pts)) # why does it return a [ [[x y]], [[x y]], ...] ?
    pts=[]; sumx=0; sumy=0
    for v in vertices:
        x=v[0][0]; y=v[0][1]
        sumx+=x; sumy+=y
        pts.append( [x,y] )
    ctrx=sumx/len(vertices); ctry=sumy/len(vertices)
    for i,v in enumerate(pts):
        x=round(ctrx+(v[0]-ctrx)*scale_factor + offset_x)
        y=round(ctry+(v[1]-ctry)*scale_factor + offset_y)
        pts[i]=(x,y)
    return pts

def IsFrameGrayScale(frame):
    if frame.dtype != np.uint8:
        sys.exit(f"cv2 returned unexpected video frame data type: {frame.dtype}")
    if len(frame.shape) == 2: return True
    if len(frame.shape) == 3:
        if frame.shape[2] == 3:
            return False
    sys.exit(f"cv2 returned unexpected video frame shape: {frame.shape}")

def ProcessOneVideo_face(input_file, output_file):
    hc=GetHaarCascade()

    cap = cv2.VideoCapture(input_file) # Open the input video
    if not cap.isOpened():
        print(f"could not open input video file '{input_file}'",file=sys.stderr)
        sys.exit(-1)
    print(cv2.__file__)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    update_interval=10

    # Define the codec (H.264 for high quality, fallback to mp4v if needed)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)
    if not out.isOpened():
        print(f"could not create output video file '{output_file}'",file=sys.stderr)
        cap.release()
        sys.exit(-2)

    print(f"extracting face mask from video file: {input_file}")
    pbar= tqdm.tqdm(total=total_frames, desc="Processing frames", unit="frame")
    frame_count = 0; isGrayScale=False
    maskImg = Image.new('L', (width,height))
    while True:
        maskImg.paste( 0, (0, 0, maskImg.size[0], maskImg.size[1])) # fill with black
        ret, frame = cap.read()
        if not ret: break

        if 0==frame_count:
            isGrayScale=IsFrameGrayScale(frame)
        frameh, framew, n = frame.shape #; print("w h extracted:",h,w,n)

        if isGrayScale:
            faces_rect = hc.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5,minSize=(64,64))
        else:
            faces_rect = hc.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5,minSize=(64,64))

        for x, y, w, h in faces_rect: #open cv Rect coordinates are [inclusive,exclusive)
            x0,y0,x1,y1=ExpandHCFaceRect(x,y,w,h,framew,frameh)
            if isGrayScale:
                imgcrop = frame[y0:y1, x0:x1]
            else:
                imgcrop = frame[y0:y1, x0:x1, :]
            #print(f"   haar cascade found face: {x0},{y0} {x1},{y1}, imgcrop shape: {imgcrop.shape}")
            #cv2.imwrite(f"facerect/{frame_count:03}_crop.png", imgcrop)
            face_landmarks_list = face_recognition.face_landmarks(imgcrop)
            if len(face_landmarks_list)>0:
                pts=_convex_hull_from_face_landmarks(face_landmarks_list[0], x0, y0, 1.05)
                ImageDraw.Draw(maskImg).polygon(pts,fill=255,outline=255)
        #for x, y, w, h in faces_rect:

        out.write(np.array(maskImg, dtype=np.uint8))
        frame_count += 1
        # Update progress bar every N frames
        if frame_count % update_interval == 0 or frame_count == total_frames:
            pbar.update( frame_count - pbar.n)

    pbar.close()
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames. Done.")

def GetFaceMasks_PIL(img):
    maskImg=Image.new('L',size=img.size,color=0)

    hc=GetHaarCascade()
    if img.mode=="L": cv_img = np.array(img)
    else: cv_img = np.array(img.convert("L"))
    faces_rect = hc.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=5,minSize=(64,64))
    if 0==len(faces_rect):
        print("no face detected")

    for x, y, w, h in faces_rect: #open cv Rect coordinates are [inclusive,exclusive)
        x0,y0,x1,y1=ExpandHCFaceRect(x,y,w,h,img.size[0],img.size[1])
        #print(f"haar cascade found face: {x0},{y0} {x1},{y1}")
        imgcrop=img.crop((x0,y0,x1,y1))

        face_landmarks_list = face_recognition.face_landmarks(pil_to_cv2(imgcrop))
        if len(face_landmarks_list)==0:
            print(f"failed to detect facial features in face at ({x0}, {y0})")
            continue

        pts=_convex_hull_from_face_landmarks(face_landmarks_list[0], x0, y0, 1.05)
        ImageDraw.Draw(maskImg).polygon(pts,fill=255,outline=255)
    return maskImg

def ProcessOneImage_face(inimage, outimage, alpha_png):
    img=func_misc.LoadInputImage(inimage)
    maskImg=GetFaceMasks_PIL(img)
    func_misc.SaveOutputImage(img, maskImg, outimage, alpha_png)
