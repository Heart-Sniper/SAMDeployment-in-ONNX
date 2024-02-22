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

<center class="half">

  <img src=README_img/point1.png /><img src=README_img/[157_66].png>
</center>

test point coordinates: [323, 541]</br>

<center class="half">

  <img src=README_img/point2.png width="300" /><img src=README_img/[323_541].png width="300">
</center>

### Bounding Box

+ test box coordinates: [116, 182, 148, 236]<br>

<center class="half">

  <img src=README_img/box1.png width="300" /><img src=README_img/[116_182_148_236].png width="300">
</center>

+ test box coordinates: [478, 930, 521, 1014]<br>

<center class="half">

  <img src=README_img/box2.png width="300" /><img src=README_img/[478_930_521_1014].png width="300">
</center>