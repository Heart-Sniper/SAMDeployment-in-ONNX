import argparse

import torch
from collections import OrderedDict
from functools import partial
from segment_anything.modeling.image_encoder import ImageEncoderViT

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=r"Model/sam_vit_b_01ec64.pth")
    parser.add_argument("--save-path", type=str)

    return parser.parse_args()


def main(args):
    encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )

    param = torch.load(args.checkpoint_path)

    d = OrderedDict()
    for k in param:
        if "image_encoder" in k:
            d[k[14:]] = param[k]

    encoder.load_state_dict(d)
    encoder.eval()

    x = torch.randn((1, 3, 1024, 1024))
    torch.onnx.export(encoder,
                    x,
                    args.save_path,
                    opset_version=12,
                    input_names=["input"],
                    output_names=["output"])


if __name__ == "__main__":
    args = parse_args()
    main(args)