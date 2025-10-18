import os, sys, click

TORCH_MODEL_FILES={'dis':'isnet-general-use.pth', 'u2net':'u2net.pth', 'u2netp':'u2netp.pth', 'u2neths':'u2net_human_seg.pth'}

@click.group()
def cli():
    """Image processing CLI tool with convert and quantize commands."""
    pass

@cli.command()
@click.argument('model_name', type=click.Choice(['u2net', 'u2netp', 'u2neths']))
@click.option('--output-file', '-o', help='Destination image file path')


def GetDefaultTorchModelPath(model_name:str):
    if model_name not in TORCH_MODEL_FILES:
        sys.exit(f"unexpected U2Net model name: {model_name}")
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir,"pretrained_models",TORCH_MODEL_FILES[model_name])

def convert(model_name, output_file):

    pth_file=GetDefaultTorchModelPath(model_name)
    if not os.path.isfile(pth_file):
        print(f"PyTorch file doesn't exist: {pth_file}",file=sys.stderr)
        quit()

    if output_file is None:
        base, _ = os.path.splitext(pth_file)
        output_file = base + ".onnx"
    print("input :",pth_file)
    print("output:",output_file)
    if os.path.isfile(output_file):
        print(f"output file already exists",file=sys.stderr)
        quit()

    import torch
    from u2net_engine import U2NET, U2NETP

    if (model_name=='u2netp'):
        net = U2NETP(3,1)
    else: # u2net or u2neths
        net = U2NET(3,1)

    mb= int(os.stat(pth_file).st_size/1048576+0.5)
    print(f"loading {pth_file}, {mb} MB, CPU mode ...")
    net.load_state_dict(torch.load(pth_file, map_location='cpu'))
    net.eval()

    input = torch.randn(1, 3, 320, 320, device='cpu')
    torch.onnx.export(net, input, output_file, export_params=True,opset_version=11,
                      do_constant_folding=True,input_names = ['input'])
    click.echo("looks like conversion finished successfully")

@cli.command()
@click.argument("input_file",type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument("output_file",type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.option("--quant-type", '-t', type=click.Choice(["0", "1"], case_sensitive=False),
    default="1", help="quantization type: 0=int8, 1=uint8 (default)" )
def quantize(input_file, output_file, quant_type):
    print("performing dynamic quantization ...")
    print("input  file:",input_file)
    print("output file:",output_file)
    #print(input_file, output_file, quant_type)

    from onnxruntime.quantization import quantize_dynamic, QuantType
    qtype=QuantType.QInt8 if quant_type=="0" else QuantType.QUInt8
    print(f"quantization type: {quant_type}/{qtype}")
    quantize_dynamic(input_file,output_file, weight_type=qtype)
    print("input file size:",round(os.path.getsize(input_file)/1048576),"MB")
    print("output file size:",round(os.path.getsize(output_file)/1048576),"MB")

if __name__ == '__main__':
    cli()
