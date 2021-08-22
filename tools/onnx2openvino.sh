# Inference Engine version: 	2021.4.0-3839-cd81789d294-releases/2021/4
# Model Optimizer version: 	2021.4.0-3839-cd81789d294-releases/2021/4

python3 mo.py --input_model psp_mobile_256p.onnx --output_dir ./openvino  --data_type FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5] --move_to_preprocess
