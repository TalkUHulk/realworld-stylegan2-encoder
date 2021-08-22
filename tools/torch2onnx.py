import torch


def stylegan2onnx(_ckpt, _onnx):
    from stylegan2.model import Generator

    # change Generator forward to forward_z
    model = Generator(256, 512, 8)
    ckpt = torch.load(_ckpt, map_location="cpu")
    model.load_state_dict(ckpt['g_ema'], strict=False)
    dlatent = torch.randn(1, 512, device='cpu')

    torch.onnx.export(
        model=model,
        args=dlatent,
        f=_onnx,
        verbose=True,
        do_constant_folding=False,
        input_names=['styles'],
        output_names=['images'],
        opset_version=10
    )


def psp2onnx(_ckpt, _onnx):
    from argparse import Namespace
    from psp.models import pSp

    ckpt = torch.load(_ckpt,
                      map_location='cpu')

    img_tensor = torch.randn(1, 3, 256, 256)

    opts = Namespace(**ckpt['opts'])
    opts.checkpoint_path = "./cartoon_psp_mobile_256p.pt"
    opts.device = "cpu"

    net = pSp(opts)
    net.eval()

    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    torch.onnx.export(
        model=net,
        args=img_tensor,
        f=_onnx,
        verbose=False,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=10
    )


def e4e2onnx(_ckpt, _onnx_encoder, _onnx_decoder):
    from e4e.encoder import Encoder4EditingMobileNet
    from stylegan2.model import Generator
    import torch
    from collections import OrderedDict

    img_tensor = torch.randn(1, 3, 256, 256)
    latents_tensor = torch.randn(1, 18, 512)

    encoder = Encoder4EditingMobileNet()
    decoder = Generator(1024, 512, 8, channel_multiplier=2)

    ckpt = torch.load(_ckpt, map_location="cpu")

    state_dict_decoder = OrderedDict()
    state_dict_encoder = OrderedDict()

    latent_avg = ckpt["latent_avg"]
    state_dict_encoder["latent_avg"] = latent_avg
    for k, v in ckpt["state_dict"].items():
        if "encoder" in k:
            state_dict_encoder[k[8:]] = v
        if "decoder" in k:
            state_dict_decoder[k[8:]] = v

    encoder.load_state_dict(state_dict_encoder)
    decoder.load_state_dict(state_dict_decoder)
    encoder.eval()
    decoder.eval()

    dynamic_axes = {'input': {0: 'batch'}, 'dlatents': {0: 'batch'}}
    torch.onnx.export(
        model=encoder,
        args=img_tensor,
        f=_onnx_encoder,
        verbose=False,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['dlatents'],
        opset_version=11
    )

    dynamic_axes = {'dlatents': {0: 'batch'}, 'output': {0: 'batch'}}
    torch.onnx.export(
        model=decoder,
        args=latents_tensor,
        f=_onnx_decoder,
        verbose=False,
        do_constant_folding=True,
        input_names=['dlatents'],
        output_names=['output'],
        opset_version=10
    )
