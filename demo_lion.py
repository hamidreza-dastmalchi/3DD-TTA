# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
    require diffusers-0.11.1
"""
import os
import clip
import torch
from PIL import Image
from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points

model_path = './lion_ckpts/epoch_10999_iters_2100999.pt'
model_config = './lion_ckpts/unconditional_all55_cfg.yml'

config.merge_from_file(model_config)
lion = LION(config)
lion.load_model(model_path)


clip_feat = None
output = lion.sample(1 if clip_feat is None else clip_feat.shape[0], clip_feat=clip_feat)
pts = output['points']
img_name = "./tmp/tmp.png"
plot_points(pts, output_name=img_name)
img = Image.open(img_name)
# img.show()
