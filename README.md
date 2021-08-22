# RealWorld StyleGan2 Encoder
  <img src="https://img.shields.io/badge/python3-pytorch-orange"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>


<p align="center">
<img src="sample/res50_age_edit_1024p.png" width="800px"/>
<br>
The demo of different style of age edit.</p>

## Description  

Various applications based on Stylegan2 Style mixing that can be inference on cpu.

## Citation

This code is heavily based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch). Thanks `rosinality` so much to make his work available 🙏🙏🙏 


## Application

<p align="center">
<img src="sample/psp_mobile_256p.png" width="800px"/>
<br>
The demo of different style of psp-mobile-256p.</p>

<p align="center">
<img src="sample/psp_res50_256p.png" width="800px"/>
<br>
The demo of different style of psp-res50-256p.</p>

<p align="center">
<img src="sample/e4e_mobile_1024p.png" width="800px"/>
<br>
The demo of different style of e4e-mobile-1024p.</p>

<p align="center">
<img src="sample/e4e_res50_1024p.png" width="800px"/>
<br>
The demo of different style of e4e-res50-1024p.</p>

<p align="center">
<img src="sample/res50_age_edit_1024p.png" width="800px"/>
<br>
<img src="sample/res50_gender_edit_1024p.png" width="800px"/>
<br>
<img src="sample/res50_pose_edit_1024p.png" width="800px"/>
<br>
<img src="sample/res50_smile_edit_1024p.png" width="800px"/>
<br>
The demo of different style edit of e4e-res50-1024p.</p>


## Pretrained Models   

I provide some of the model to test.
| Path | Description
| :--- | :----------
|[torch weight](https://pan.baidu.com/s/1Gq0nV2Fn1jHhE-HQ9NhJEQ) | cartoon of e4e-mbv3 1024p and psp-mbv3 256p 密码: 0mgk
|[onnx weight](https://pan.baidu.com/s/1jXv3aQX0wEDOjCWi_1seyg) | stylegan2 onnx, cartoon of e4e-mbv3 1024p and psp-mbv3 256p 密码: inn8
|[openvino weight](https://pan.baidu.com/s/15w4gTOUbcjHw_cz9FGtXAw) | openvino cartoon of e4e-mbv3 1024p and psp-mbv3  密码: 759q

## Comparison
*All test on MacOs 11.4 | 2.6 GHz Intel Core i7 | 32 GB 2667 MHz DDR4*

**pSp-mbv3-256p**
|Name|Time(s)|
|:---|:--|
|torch|1.9860|
|onnx|0.5869|
|openvino|0.4533|

**e4e-mbv3-1024p**
|Name|Time(s)|
|:---|:--|
|  torch |4.8058|
|   onnx |1.5155|
|openvino|0.8690|

## Test
### torch
```
python scripts/test.py \
--ckpt ./best_model.pt \
--network psp \
--platform torch \
--align \
--images_path ./test_images
```

### onnx
```
python scripts/test.py \
--ckpt ./cartoon_psp_mobile_256p.onnx \
--network psp \
--platform onnx \
--align \
--images_path ./test_images
```

### openvino
``` 
python scripts/test.py \
--ckpt_encoder ./art_mobile_encoder_1024p \
--ckpt_decoder ./art_decoder_1024p \
--network e4e \
--platform openvino \
--align \
--images_path ./test_images \
--edit \
--edit_direction ./editings/smile.npy
```

## Convert
### torch2onnx
``` 
function can find in tools/torch2onnx.py
```

### onnx2openvino
*Inference Engine version: 	2021.4.0-3839-cd81789d294-releases/2021/4*
*Model Optimizer version: 	2021.4.0-3839-cd81789d294-releases/2021/4*

```
python3 mo.py \
--input_model psp_mobile_256p.onnx \
--output_dir ./openvino  \
--data_type FP16 \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5,127.5,127.5] \
--move_to_preprocess
```