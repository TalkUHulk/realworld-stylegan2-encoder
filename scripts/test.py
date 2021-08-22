import argparse
from argparse import Namespace
import os
import torch
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(".")
sys.path.append("..")


def test_onnx(args):
    import onnxruntime as ort
    from tools.common import normalization
    from PIL import Image

    if args.network == "stylegan2":
        ort_session = ort.InferenceSession(args.ckpt)
        for i in tqdm(range(args.pics), desc="stylegan2 generating..."):
            latents = np.random.randn(1, 512).astype(np.float32)
            outputs = ort_session.run([ort_session.get_outputs()[0].name],
                                      {ort_session.get_inputs()[0].name: latents})[0]

            output = (outputs.squeeze().transpose((1, 2, 0)) + 1) / 2
            output[output < 0] = 0
            output[output > 1] = 1
            output = normalization(output) * 255
            output = Image.fromarray(output.astype('uint8'))
            output.save(os.path.join(args.save, "{}_{}_{:05d}.png".format(args.network, args.platform, i)))

    elif args.network == "psp":
        from PIL import Image

        images = [x.path for x in os.scandir(args.images_path) if x.name.endswith(("png", "jpg", "jpeg"))]

        ort_session = ort.InferenceSession(args.ckpt)

        for i, image in tqdm(enumerate(images), desc="stylegan2 generating..."):
            if args.align:
                im = detect_and_align_face(image)
            else:
                im = Image.open(image)

            img = (np.array(im).transpose((2, 0, 1)) - 127.5) / 127.5
            img = np.expand_dims(img, 0).astype(np.float32)
            output = ort_session.run([ort_session.get_outputs()[0].name],
                                     {ort_session.get_inputs()[0].name: img})[0]

            output = (output.squeeze().transpose((1, 2, 0)) + 1) / 2
            output[output < 0] = 0
            output[output > 1] = 1
            output = normalization(output) * 255
            output = Image.fromarray(output.astype('uint8'))
            output.save(os.path.join(args.save, "{}_{}_{}.png".format(
                args.network,
                args.platform,
                image.split("/")[-1].split(".")[0])))

    elif args.network == "e4e":
        from PIL import Image
        images = [x.path for x in os.scandir(args.images_path) if x.name.endswith(("png", "jpg", "jpeg"))]

        ort_session_encoder = ort.InferenceSession(args.ckpt_encoder)
        ort_session_decoder = ort.InferenceSession(args.ckpt_decoder)

        for i, image in tqdm(enumerate(images), desc="stylegan2 generating..."):
            if args.align:
                im = detect_and_align_face(image)
            else:
                im = Image.open(image)

            img = (np.array(im).transpose((2, 0, 1)) - 127.5) / 127.5
            img = np.expand_dims(img, 0).astype(np.float32)

            dlatents = ort_session_encoder.run([ort_session_encoder.get_outputs()[0].name],
                                               {ort_session_encoder.get_inputs()[0].name: img})[0]

            if args.edit:
                edit_direction = np.load(args.edit_direction)
                dlatents += edit_direction * args.edit_alpha

            output = ort_session_decoder.run([ort_session_decoder.get_outputs()[0].name],
                                             {ort_session_decoder.get_inputs()[0].name: dlatents})[0]

            output = (output.squeeze().transpose((1, 2, 0)) + 1) / 2
            output[output < 0] = 0
            output[output > 1] = 1
            output = normalization(output) * 255
            output = Image.fromarray(output.astype('uint8'))
            output.save(os.path.join(args.save, "{}_{}_{}.png".format(
                args.network,
                args.platform,
                image.split("/")[-1].split(".")[0])))

    else:
        print("network:{} not support, only support stylegan2、psp or e4e.".format(args.network))


def test_openvino(args):
    from openvino.inference_engine import IECore
    from tools.common import normalization
    from PIL import Image
    if args.network == "stylegan2":
        ie = IECore()
        net = ie.read_network(model=args.ckpt + ".xml", weights=args.ckpt + ".bin")
        input_blob = list(net.input_info.keys())[0]
        exec_net = ie.load_network(net, "CPU")

        for i in tqdm(range(args.pics), desc="stylegan2 generating..."):
            latents = np.random.randn(1, 512).astype(np.float32)
            res = exec_net.infer(inputs={input_blob: [latents]})
            outputs = res["images"].squeeze()
            output = (outputs.squeeze().transpose((1, 2, 0)) + 1) / 2
            output[output < 0] = 0
            output[output > 1] = 1
            output = normalization(output) * 255
            output = Image.fromarray(output.astype('uint8'))
            output.save(os.path.join(args.save, "{}_{}_{:05d}.png".format(args.network, args.platform, i)))

    elif args.network == "psp":
        images = [x.path for x in os.scandir(args.images_path) if x.name.endswith(("png", "jpg", "jpeg"))]

        ie = IECore()
        net = ie.read_network(model=args.ckpt + ".xml", weights=args.ckpt + ".bin")
        input_blob = list(net.input_info.keys())[0]
        exec_net = ie.load_network(net, "CPU")

        for i, image in tqdm(enumerate(images), desc="stylegan2 generating..."):
            if args.align:
                im = detect_and_align_face(image)
            else:
                im = Image.open(image)

            img = np.array(im).transpose((2, 0, 1))
            res = exec_net.infer(inputs={input_blob: [img]})
            output = res["output"].squeeze()

            output = normalization(output)
            output = np.transpose(np.clip(np.multiply(output, 255) + 0.5, 0, 255), (1, 2, 0)).astype(np.uint8)
            output = Image.fromarray(output)
            output.save(os.path.join(args.save, "{}_{}_{:05d}.png".format(args.network, args.platform, i)))

    elif args.network == "e4e":
        images = [x.path for x in os.scandir(args.images_path) if x.name.endswith(("png", "jpg", "jpeg"))]

        ie_encoder = IECore()
        encoder = ie_encoder.read_network(model=args.ckpt_encoder + ".xml", weights=args.ckpt_encoder + ".bin")
        encoder_input_blob = list(encoder.input_info.keys())[0]
        exec_encoder = ie_encoder.load_network(encoder, "CPU")

        ie_decoder = IECore()
        decoder = ie_decoder.read_network(model=args.ckpt_decoder + ".xml", weights=args.ckpt_decoder + ".bin")
        decoder_input_blob = list(decoder.input_info.keys())[0]
        exec_decoder = ie_decoder.load_network(decoder, "CPU")

        for i, image in tqdm(enumerate(images), desc="stylegan2 generating..."):
            if args.align:
                im = detect_and_align_face(image)
            else:
                im = Image.open(image)

            img = np.array(im).transpose((2, 0, 1))

            res = exec_encoder.infer(inputs={encoder_input_blob: [img]})
            dlatents = res["dlatents"]

            if args.edit:
                edit_direction = np.load(args.edit_direction)
                dlatents += edit_direction * args.edit_alpha

            res = exec_decoder.infer(inputs={decoder_input_blob: [dlatents]})
            output = res["output"].squeeze()

            output = normalization(output)
            output = np.transpose(np.clip(np.multiply(output, 255) + 0.5, 0, 255), (1, 2, 0)).astype(np.uint8)
            output = Image.fromarray(output)
            output.save(os.path.join(args.save, "{}_{}_{:05d}.png".format(args.network, args.platform, i)))
    else:
        print("network:{} not support, only support stylegan2、psp or e4e.".format(args.network))


@torch.no_grad()
def test_torch(args):
    from stylegan2.model import Generator

    if args.network == "stylegan2":
        from torchvision import utils
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        )
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
        g_ema.eval()
        if args.truncation < 1:
            mean_latent = g_ema.mean_latent(args.truncation_mean)
        else:
            mean_latent = None

        for i in tqdm(range(args.pics), desc="stylegan2 generating..."):
            sample_z = torch.randn(args.sample, args.latent)

            sample, _ = g_ema.forward_ori(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                os.path.join(args.save, "{}_{}_{:05d}.png".format(args.network, args.platform, i)),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

    elif args.network == "psp":
        from PIL import Image
        from psp.models import pSp
        import torchvision.transforms as transforms
        from torchvision import utils
        from collections import OrderedDict
        from tools.common import trans

        images = [x.path for x in os.scandir(args.images_path) if x.name.endswith(("png", "jpg", "jpeg"))]
        ckpt = torch.load(args.ckpt, map_location='cpu')
        opts = Namespace(**ckpt['opts'])
        opts.checkpoint_path = args.ckpt
        opts.device = "cpu"
        net = pSp(opts)
        net.eval()
        for i, image in tqdm(enumerate(images), desc="stylegan2 generating..."):
            if args.align:
                im = detect_and_align_face(image)
            else:
                im = Image.open(image)

            img = net(trans(im).unsqueeze(0))
            utils.save_image(
                img,
                os.path.join(args.save, "{}_{}_{}.png".format(
                    args.network,
                    args.platform,
                    image.split("/")[-1].split(".")[0])),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

    elif args.network == "e4e":
        from PIL import Image
        from e4e.encoder import Encoder4EditingMobileNet
        from stylegan2.model import Generator
        from torchvision import utils
        from tools.common import trans
        from collections import OrderedDict

        images = [x.path for x in os.scandir(args.images_path) if x.name.endswith(("png", "jpg", "jpeg"))]

        ckpt = torch.load(args.ckpt, map_location='cpu')
        encoder = Encoder4EditingMobileNet()
        decoder = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier)

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

        for i, image in tqdm(enumerate(images), desc="stylegan2 generating..."):
            if args.align:
                im = detect_and_align_face(image)
            else:
                im = Image.open(image)

            dlatents = encoder(trans(im).unsqueeze(0))
            img = decoder(dlatents).squeeze(0)

            utils.save_image(
                img,
                os.path.join(args.save, "{}_{}_{}.png".format(
                    args.network,
                    args.platform,
                    image.split("/")[-1].split(".")[0])),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
    else:
        print("network:{} not support, only support stylegan2、psp or e4e.".format(args.network))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--network", type=str, default="stylegan2", help="type of model, [stylegan2/psp/e4e]"
    )

    parser.add_argument(
        "--platform", type=str, default="torch", help="inference platform, [torch/onnx/openvino]"
    )

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=8, help="number of images to be generated"
    )
    parser.add_argument(
        "--images_path", type=str, help="images path to be test."
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )

    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    parser.add_argument(
        "--save",
        type=str,
        default="./sample",
        help="path to save the result",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to the model checkpoint, if openvino, ckpt is the openvino bin except extensions",
    )

    parser.add_argument(
        "--ckpt_encoder",
        type=str,
        help="path to the model checkpoint of e4e encoder, if openvino, ckpt is the openvino bin except extensions",
    )

    parser.add_argument(
        "--ckpt_decoder",
        type=str,
        help="path to the model checkpoint of e4e decoder, if openvino, ckpt is the openvino bin except extensions",
    )

    parser.add_argument(
        '--align',
        action="store_true",
        help='Whether to use face align.')

    parser.add_argument(
        '--edit',
        action="store_true",
        help='Whether to edit when test e4e.')

    parser.add_argument(
        '--edit_direction',
        type=str,
        help='edit direction npy file.')

    parser.add_argument(
        '--edit_alpha',
        type=float,
        default=1.0,
        help='edit direction alpha.')

    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.align:
        from tools.common import detect_and_align_face

    if args.platform == "torch":
        test_torch(args)
    elif args.platform == "onnx":
        test_onnx(args)
    elif args.platform == "openvino":
        test_openvino(args)
    else:
        print("platform:{} not support, only support torch、onnx or openvino.".format(args.platform))

    print("Test Finished!")

