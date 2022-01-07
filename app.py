'''
Author: Egrt
Date: 2022-01-04 21:46:25
LastEditors: Egrt
LastEditTime: 2022-01-07 19:49:19
FilePath: \License-super-resolution-master\app.py
'''
from Utilities.io import DataLoader
from Models.RRDBNet import RRDBNet
import numpy as np
import gradio as gr
import cv2
import os
loader = DataLoader()
# --------加载模型---------- #
MODEL_PATH = 'Pretrained/rrdb'
model = RRDBNet(blockNum=10)
model.load_weights(MODEL_PATH)

# --------模型推理---------- #
def inference(file):
    # 将np转Tensor
    input_image= loader.input_image(file.name)
    # 维度扩张
    input_image= np.expand_dims(input_image, axis=0)
    yPred      = model.predict(input_image)
    yPred      = np.squeeze(np.clip(yPred, a_min=0, a_max=1))
    return yPred

# --------网页信息---------- #  
title = "车牌超分辨率"
description = "基于生成对抗网络的车牌超分辨率，可从24×12像素的超低分辨率车牌图片恢复到正常可视状态@西南科技大学智能控制与图像处理研究室"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2108.10257' target='_blank'>SwinIR: Image Restoration Using Swin Transformer</a> | <a href='https://github.com/JingyunLiang/SwinIR' target='_blank'>Github Repo</a></p>"
example_img_dir  = 'Samples'
example_img_name = os.listdir(example_img_dir)
examples=[[os.path.join(example_img_dir, image_path)] for image_path in example_img_name if image_path.endswith('.jpg')]
gr.Interface(
    inference, 
    [gr.inputs.Image(type="file", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    title=title,
    description=description,
    article=article,
    enable_queue=True,
    examples=examples
    ).launch(debug=True)
