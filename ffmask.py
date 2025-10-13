# minimal imports so if command line was not entered correctly,
# we don't incur long startup time of heavy modules
import os, sys, click

opt_exec_provider = click.option('-e', '--exec-provider', type=str, default='', help='set ONNX execution provider, can be partial name eg cpu, cuda')
opt_detection_model = click.option('-m', '--model', type=click.Choice(['dis','u2net', 'u2netp', 'u2neths', 'face']),
                                    default='u2net', show_default=True, help='select model to use')
arg_input_file = click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
arg_output_file = click.argument("output_file", type=click.Path(exists=False,file_okay=True, dir_okay=False, writable=True))

@click.group()
@click.version_option(version="2.0")

def cli():
    """Extract mask of foreground object or face in image or video"""
    pass

@cli.command()
@arg_input_file
@arg_output_file
@opt_detection_model
@opt_exec_provider
def video(input_file, output_file, model, exec_provider):
    """Process an input video and save to output video in MP4 format."""
    if model=='face':
        import func_face
        func_face.ProcessOneVideo_face(input_file, output_file)
    else:
        import func_onnx
        func_onnx.ProcessOneVideo(input_file, output_file, model, exec_provider)

@cli.command()
@arg_input_file
@arg_output_file
@opt_detection_model
@opt_exec_provider
@click.option('-a', '--alpha-png', is_flag=True, default=False, help='save detected mask as alpha channel in PNG file')
def image(input_file, output_file, model, exec_provider, alpha_png):
    """Process an input image and save to output image"""
    if alpha_png==True: # if saving PNG file
        if os.path.splitext(output_file)[1].lower() != ".png":
            sys.exit("ERROR: alpha-png option specified but output file does not have .png extension")

    if model=='face':
        import func_face
        func_face.ProcessOneImage_face(input_file, output_file, alpha_png)
    else:
        import func_onnx
        func_onnx.ProcessOneImage(input_file, output_file, model, exec_provider, alpha_png)

@cli.command()
def listep():
    """List installed ONNX execution providers."""
    import onnxruntime as ort
    eps = ort.get_available_providers()
    print("available ONNX execution providers:")
    for provider in eps:
        print(" * " + provider)

if __name__ == '__main__':
    cli()

