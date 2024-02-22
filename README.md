# Segment Anything Deployment with ONNXRuntime C++

## Requirements

  + OnnxRuntime.GPU.1.15.1
  + OpenCV 4.8

## Convert to ONNX

### Image Encoder

To convert the image encoder to ONNX, try to use the following script:

```shell
!python Export2ONNX\convert2onnx.py \
    --checkpoint-path Model\sam_vit_b_01ec64.pth \
    --save-path Model\sam_vit_b_imgencoder.onnx
```
Note: the default image size is [1024，1024]，you can customize image size and other parameters.

### Prompt Encoder & Mask Decoder

You can easily convert both prompt encoder and mask decoder to ONNX through official script in `segment-anything` package.</br>
Here is an example:

```shell
!python segment-anything\scripts\export_onnx_model.py \
    --checkpoint Model\sam_vit_b_01ec64.pth \
    --output Model\sam_vit_b_pmencoder.onnx \
    --model-type vit_b \
    --opset 12
```

## Deployment

coming soon...

## Performance

Note: the following example is inferenced based on the base model. 

### Key Point

test point coordinates: [157, 66]</br>

![img](README_img/[157_66].png)

test point coordinates: [323, 541]</br>

![img](README_img/[323_541].png)

### Bounding Box

test box coordinates: [116, 182, 148, 236]<br>

![img](README_img/[116_182_148_236].png)

test box coordinates: [478, 930, 521, 1014]<br>

![img](README_img/[478_930_521_1014].png)