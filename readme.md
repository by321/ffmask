## ffmask - detect foreground object or human face from images and videos

This is a Python project (both CLI and GUI) based on [DIS](https://github.com/xuebinqin/DIS "DIS"), [U²-Net](https://github.com/xuebinqin/U-2-Net "u2net"), and [Face Recognition](https://github.com/ageitgey/face_recognition "Face Recognition"):

- Use DIS or u2net used to detect foreground objects. For u2net, the default u2net model, the small and fast version (u2netp), and human segmentation version are supported.
- Use face recognition to detect face outline.
- Can process both image and videos.
- Run on either CPU or GPU (using ONNX runtime).

![ffmask GUI](images/ffmask_gui.png)

![boat](images/boat.jpg)
![water drop](images/waterdrop.jpg)

Face detection mode (the "-m face" or "--model face" option):

![Obama Trump](images/obamatrump.png)

## Installation

- Python version 3.9 or later. Create a virtual environment if you want to.
- Install ONNX runtime for your system.
- Clone this repository.
- Run "pip install -r requirements.txt".
- Download the model files from https://huggingface.co/by321/ffmask/tree/main and save to pretrained_models directory:

	- haarcascade_frontalface_alt2.xml : for Haar Cascasde face detection
	- haarcascade_frontalface_default.xml : for Haar Cascade face detection
	- isnet-general-use.pth.onnx : Dichotomous Image Segmentation model file
	- u2net.pth.onnx : u2net model file
	- u2netp.pth.onnx : small and fast version of u2net
	- u2net_human_seg.pth.onnx : u2net variant trained for human detection

 If you see an error message about missing openh264-*.dll, download the appropriate DLL from https://github.com/cisco/openh264 and put it somewhere on your path.


## GUI Usage

Run "python ffmask_gui.py", wait until you see a message like "Running on local URL: http://127.0.0.1:7860". Open a web browser, type "127.0.0.1:7860" in the address bar and press ENTER. You should see the GUI page. Here's the general usage:

- Drag and drop an image onto input image box, select a model, click on "Create Mask".
- You can then select filters and click on "Filter Mask" to apply filters to the selected mask. A new, filtered mask will be generated.
- The combined view image box lets you see the masked input image (input*mask/255, basically using mask as an alpah channel), mask image, and input image. All these images can be downloaded.

The GUI is built using gradio version 5.44.1.

## CLI Usage

Run "python ffmask.py" will print an overview of the usage:

	Usage: ffmask.py [OPTIONS] COMMAND [ARGS]...

	Extract mask of foreground object or face in image or video

	Options:
	--version  Show the version and exit.
	--help     Show this message and exit.

	Commands:
	image   Process an input image and save to output image
	listep  List installed ONNX execution providers.
	video   Process an input video and save to output video in MP4 format.

Run "python ffmask.py image --help" or "python ffmask.py video --help" will show further usage text.

Usage examples:

    python ffmask.py image input.jpg mask.jpg

    python ffmask.py video input.mp4 mask.mp4

A mask is generated for input image or each video frame. No further processing of the mask is done.
The intention is to use the mask in image or video editors for further processing.

## Creating and Quantizing ONNX model files

The ONNX models files were converted from original PyTorch files using conv_u2net_to_onnx.py. You can also use conv_u2net_to_onnx.py to truncate ONNX files for smaller model size and faster execution time.
Run "python conv_u2net_to_onnx.py --help" for usage info.

