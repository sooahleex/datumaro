import os
import numpy as np
import math
from PIL import Image
import torch
from torchvision import transforms
from efficientnet import EfficientNet
from tqdm import tqdm

from torch.autograd import Function

device = "cuda" if torch.cuda.is_available() else "cpu"
model_ = {'effb0': None}

def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).
    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.
    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels

def get_efficientnet(
    version,
    in_size,
    tf_mode=False,
    bn_eps=1e-5,
    model_name=None,
    pretrained=False,
    root=os.path.join("~", ".torch", "models"),
    **kwargs,
):
    """
    Create EfficientNet model with specific parameters.
    Parameters:
    ----------
    version : str
        Version of EfficientNet ('b0'...'b8').
    in_size : tuple of two ints
        Spatial size of the expected input image.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if version == "b0":
        assert in_size == (224, 224)
        depth_factor = 1.0
        width_factor = 1.0

    init_block_channels = 32
    layers = [1, 2, 2, 3, 3, 4, 1]
    downsample = [1, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
    expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
    strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
    final_block_channels = 1280

    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

    from functools import reduce

    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [],
    )
    kernel_sizes = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(kernel_sizes_per_layers, layers, downsample),
        [],
    )
    expansion_factors = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(expansion_factors_per_layers, layers, downsample),
        [],
    )
    strides_per_stage = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(strides_per_stage, layers, downsample),
        [],
    )
    strides_per_stage = [si[0] for si in strides_per_stage]

    init_block_channels = round_channels(init_block_channels * width_factor)

    if width_factor > 1.0:
        assert int(final_block_channels * width_factor) == round_channels(final_block_channels * width_factor)
        final_block_channels = round_channels(final_block_channels * width_factor)

    net = EfficientNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        tf_mode=tf_mode,
        bn_eps=bn_eps,
        in_size=in_size,
        **kwargs,
    )
    # model_path = './model_paths/efficientnet_b0_imagenet_cls.pth'
    # model_dict = torch.load(model_path, map_location='cpu')
    # model_dict.pop('output.fc.weight', None)
    # model_dict.pop('output.fc.bias', None)

    # model_path = './caltech101_effb0_100_trained.pth'
    model_path = './model_paths/food101_effb0_100_trained.pth'
    model_dict = torch.load(model_path, map_location='cpu')['model']['state_dict']
    model_dict.pop('output.asl.weight', None)
    model_dict.pop('output.asl.bias', None)

    print(f'load model path : {model_path}.....')
    net.load_state_dict(model_dict, device)
    net = net.to(device)
    net.eval()

    return net


def efficientnet_b0(in_size=(224, 224), **kwargs):
    """
    EfficientNet-B0 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(version="b0", in_size=in_size, model_name="efficientnet_b0", **kwargs)

class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output

def hash_layer(input):
    return hash.apply(input)

def encode_discrete(x):
    prob = torch.sigmoid(x)
    z = hash_layer(prob - 0.5)
    return z

def effb0_infer(data):
    model = model_['effb0']
    if not model:
        model = efficientnet_b0()
        model_['effb0'] = model

    trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
    ]
    )

    img = np.uint8(data)
    img = Image.fromarray(img)

    if np.array(img).ndim == 2 or np.array(img).ndim == 4:
        img = img.convert('RGB')

    img = trans(img)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    img = img.to(device, dtype=torch.float)

    features = model.forward(img)

    # features = encode_discrete(features)

    return features

import torch.nn.functional as F
def compute_hash(features):
    # features = features.view(features.shape[0], -1)
    features = encode_discrete(features)
    features = F.normalize(features, dim=-1)

    features = features.cpu()
    hash_key = features.detach().numpy() >= 0
    hash_key = hash_key*1
    hash_key = np.packbits(hash_key, axis=-1)
    return hash_key

def effb0_hash(dataset):
    database_features = None
    labels = []
    item_list = []
    database_hashes = None
    for datasetitem in tqdm(dataset):
        # hash_key = effb0_infer(datasetitem.media.data)[0].detach().numpy()
        hash_key = effb0_infer(datasetitem.media.data)[0]
        hash_bit = compute_hash(hash_key)
        if database_features is None:
            database_features = hash_key.cpu().detach().numpy().reshape(1, -1)
        else:
            database_features = np.concatenate((database_features, hash_key.cpu().detach().numpy().reshape(1, -1)), axis=0)
        if database_hashes is None:
            database_hashes = hash_bit.reshape(1, -1)
        else:
            database_hashes = np.concatenate((database_hashes, hash_bit.reshape(1, -1)), axis=0)
        labels.append(datasetitem.annotations[0].label)
        item_list.append(datasetitem)
    labels = np.array(labels)
    # return database_features, database_hashes, labels, item_list
    return database_hashes, labels, item_list