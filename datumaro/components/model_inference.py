# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import numpy as np

import torch
from torchvision import transforms
import torch.nn.functional as F

import clip

device = "cpu"
model_folder = "./tests/assets/searcher"


def load_model():
    model, _ = clip.load("ViT-B/32", device=device, jit=True)
    # model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def inference(image):
    model = load_model()

    trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
        ]
    )

    img = np.uint8(image.data)
    img = trans(img)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    img = img.to(device, dtype=torch.float)

    img_features = model.encode_image(img)
    img_features = F.normalize(img_features, dim=-1)

    return img_features


def save_hash(path: str, hash: str):
    dst_dir = osp.dirname(path)
