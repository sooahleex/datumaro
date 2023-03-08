# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import os
import logging as log
import pickle

import numpy as np
from typing import Union, List
import random
import cv2
from sklearn.cluster import KMeans
from openvino.inference_engine import IECore
import math
from tokenizers import Tokenizer
from tqdm import tqdm

from datumaro.components.dataset import IDataset
from datumaro.components.annotation import LabelCategories, Label

device = 'cpu'
model_folder = "./searcher"
model = {
    "IMG": None,
    "TXT": None,
}
tokenizer_ = {'clip-vit-base-patch320': None}

def img_center_crop(image, size):
    """
    Crop center of image
    """

    width, height = image.shape[1], image.shape[0]
    mid_w, mid_h = int(width / 2), int(height / 2)

    crop_w = size if size < image.shape[1] else image.shape[1]
    crop_h = size if size < image.shape[0] else image.shape[0]
    mid_cw, mid_ch = int(crop_w / 2), int(crop_h / 2)

    cropped_image = image[mid_h - mid_ch: mid_h +
                          mid_ch, mid_w - mid_cw: mid_w + mid_cw]
    return cropped_image


def img_normalize(image):
    mean = 255 * np.array([0.485, 0.456, 0.406])
    std = 255 * np.array([0.229, 0.224, 0.225])

    image = image.transpose(-1, 0, 1)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image


def normalize(x, axis=1, eps=1e-12):
    denom = max(np.linalg.norm(x, axis=axis, keepdims=True), eps)
    return x / denom


def _image_features(model, inputs, input_blob=None, output_blob=None):
    if np.array(inputs).ndim == 2:
        inputs = cv2.cvtColor(inputs, cv2.COLOR_GRAY2RGB)
    elif np.array(inputs).ndim == 4:
        inputs = cv2.cvtColor(inputs, cv2.COLOR_RGBA2RGB)
    inputs = cv2.resize(inputs, (256, 256))
    inputs = img_center_crop(inputs, 224)
    inputs = img_normalize(inputs)
    inputs = np.expand_dims(inputs, axis=0)

    h = model.infer(inputs={input_blob: inputs})
    features = h[output_blob]
    return features


def _compute_hash(features):
    features = np.sign(features)
    hash_key = np.clip(features, 0, None)
    hash_key = hash_key.astype(np.uint8)
    hash_key = np.packbits(hash_key, axis=-1)
    return hash_key


def hash_inference(item):
    img_xml_model_path = os.path.join(model_folder, "clip_visual_ViT-B_32.xml")
    img_bin_model_path = os.path.join(model_folder, "clip_visual_ViT-B_32.bin")
    ie = IECore()

    img_model = model["IMG"]
    if not img_model:
        img_net = ie.read_network(img_xml_model_path, img_bin_model_path)
        img_model = ie.load_network(network=img_net, device_name="CPU")
        model["IMG"] = img_model
    input_blob = next(iter(img_model.input_info))
    output_blob = next(iter(img_model.outputs))

    features = _image_features(img_model, item, input_blob, output_blob)

    hash_string = _compute_hash(features)
    return hash_string


def hash_inference_text(item):
    txt_xml_model_path = os.path.join(model_folder, "clip_text_ViT-B_32.xml")
    txt_bin_model_path = os.path.join(model_folder, "clip_text_ViT-B_32.bin")
    ie = IECore()

    txt_model = model["TXT"]
    if not txt_model:
        txt_net = ie.read_network(txt_xml_model_path, txt_bin_model_path)
        txt_model = ie.load_network(network=txt_net, device_name="CPU")
        model["TXT"] = txt_model
    input_blob = next(iter(txt_model.input_info))
    output_blob = next(iter(txt_model.outputs))

    h = txt_model.infer(inputs={input_blob: item})
    features = h[output_blob]
    hash_string = _compute_hash(features)
    return hash_string


def tokenize_list(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True):
    if isinstance(texts, str):
        texts = [texts]

    tokenizer = tokenizer_['clip-vit-base-patch320']
    if not tokenizer:
        tokenizer = Tokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer_['clip-vit-base-patch320'] = tokenizer
    tokens = [tokenizer.encode(text).ids for text in texts]
    result = np.zeros((1, context_length))
    n = 0
    for i, token in enumerate(tokens):
        if len(token) > context_length:
            if truncate:
                eot_token = token.ids[-1]
                token = token[:context_length]
                token[-1] = eot_token

        for j, token_ in enumerate(token):
            result[:, n:n+j] = token_
            n += 1
    return result


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - B1 @ B2.transpose())
    return distH


def tokenize(texts: str, context_length: int = 77, truncate: bool = True):
    tokenizer = tokenizer_['clip-vit-base-patch320']
    if not tokenizer:
        tokenizer = Tokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer_['clip-vit-base-patch320'] = tokenizer
    tokens = tokenizer.encode(texts).ids
    result = np.zeros((1, context_length))

    if len(tokens) > context_length:
        if truncate:
            eot_token = tokens[-1]
            tokens = tokens[:context_length]
            tokens[-1] = eot_token

    for i, token in enumerate(tokens):
        result[:, i] = token
    return result


cifar10_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

cifar100_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

caltech101_templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.',
]

lgchem_templates = [
    'a photo of a {}.',
    'a bad photo of the {}.',
    'a photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.'
]

coop_template = [
    'X X X X X X X X X X X X X X X X {}.'
]

svhn_templates = [
    'a photo of house address number: "{}".'
] #    'a photo of the number: "{}".',

class Prune():
    def __init__(self, dataset: IDataset, ratio_list: List[float], cluster_method: str, data_name: str, hash_type: str = None, hash_base_model:str = None) -> None:
        """

        """
        self._dataset = dataset
        self._ratio_list = ratio_list
        self._cluster_method = cluster_method
        self._data_name = data_name
        self._hash_type = hash_type

        database_keys = None
        item_list = []
        exception_items = []
        labels = []

        if self._cluster_method == 'random':
            for datasetitem in self._dataset:
                item_list.append(datasetitem)
        else:
            try:
                with open(f'./hash_pickles/{self._data_name}_{hash_base_model}_{self._hash_type}.pickle', 'rb') as handle:
                    saved_dict = pickle.load(handle)
                    database_keys = saved_dict['database_keys']
                    labels = saved_dict['labels']
                    item_list = saved_dict['item_list']
            except:
                if self._hash_type == 'img_txt':
                    category_dict = {}
                    for label, indice in list(self._dataset.categories().values())[0]._indices.items():
                        category_dict[indice] = f'a photo of {label}'
                elif self._hash_type == 'img_txt_prompt':
                    category_dict = {}
                    for label, indice in list(self._dataset.categories().values())[0]._indices.items():
                        if self._data_name == 'cifar10':
                            category_dict[indice] = [template.format(label) for template in cifar10_templates]
                        elif self._data_name == 'cifar100':
                            category_dict[indice] = [template.format(label) for template in cifar100_templates]
                        elif self._data_name == 'caltech101':
                            category_dict[indice] = [template.format(label) for template in caltech101_templates]
                        elif self._data_name == 'lgchem':
                            category_dict[indice] = [template.format(label) for template in lgchem_templates]
                        elif self._data_name == 'svhn':
                            category_dict[indice] = [template.format(label) for template in svhn_templates]
                elif self._hash_type == 'img_txt_coop':
                    category_dict = {}
                    for label, indice in list(self._dataset.categories().values())[0]._indices.items():
                        category_dict[indice] = [template.format(label) for template in coop_template]

                for datasetitem in tqdm(self._dataset):
                    try:
                        if self._hash_type in ['img_txt', 'img_txt_prompt', 'img_txt_coop']:
                            hash_key_img = hash_inference(datasetitem.media.data)[0]
                            prompt = category_dict.get(datasetitem.annotations[0].label)
                            if isinstance(prompt, List):
                                prompt = (" ").join(prompt)
                            inputs = tokenize(prompt)
                            hash_key_txt = hash_inference_text(inputs)[0]
                            hash_key = np.concatenate([hash_key_img, hash_key_txt])
                        else:
                            hash_key = hash_inference(datasetitem.media.data)[0]
                        hash_key = np.unpackbits(hash_key, axis=-1)

                        if database_keys is None:
                            database_keys = hash_key.reshape(1, -1)
                        else:
                            database_keys = np.concatenate(
                                (database_keys, hash_key.reshape(1, -1)), axis=0)
                        item_list.append(datasetitem)
                        labels.append(datasetitem.annotations[0].label)
                    except Exception:
                        exception_items.append(datasetitem)
                save_dict = {'database_keys': database_keys, 'item_list': item_list, 'labels': labels}
                with open(f'./hash_pickles/{self._data_name}_{hash_base_model}_{self._hash_type}.pickle', 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'{self._data_name}_{hash_base_model}_{self._hash_type}.pickle saved.......')

            self._database_keys = database_keys
            self._item_list = item_list
            self._exception_items = exception_items


    def get_pruned(self) -> None:
        if self._cluster_method == 'random':
            for i, ratio in enumerate(self._ratio_list):
                self._ratio = ratio

                dataset_len = len(self._item_list)
                removed_items = []
                num_selected_item = math.ceil(dataset_len*self._ratio)
                random_list = random.sample(range(dataset_len), num_selected_item)
                removed_items = list(range(dataset_len))
                for idx in random_list:
                    removed_items.remove(idx)
                removed_items = (np.array(self._item_list)[removed_items]).tolist()

                if i == 0:
                    removed_items_1 = removed_items
                elif i == 1:
                    removed_items_2 = removed_items
                elif i == 2:
                    removed_items_3 = removed_items
                elif i == 3:
                    removed_items_4 = removed_items
                elif i == 4:
                    removed_items_5 = removed_items
                elif i == 5:
                    removed_items_6 = removed_items
            return removed_items_1, removed_items_2, removed_items_3, removed_items_4, removed_items_5, removed_items_6

        for i, ratio in enumerate(self._ratio_list):
            self._ratio = ratio

            if self._cluster_method == 'centroid':
                num_centers = math.ceil(len(self._database_keys)*self._ratio)
                kmeans = KMeans(n_clusters=num_centers, random_state=0)

            elif self._cluster_method in ['prune_close', 'clustered_random']:
                if self._data_name in ['coco', 'bccd', 'pcd', 'cifar10', 'cifar100', 'svhn', 'caltech101', 'lgchem']:
                    for category in self._dataset.categories().values():
                        if isinstance(category, LabelCategories):
                            num_centers = len(list(category._indices.keys()))
                else:
                    num_centers = len(self._dataset.subsets())
                kmeans = KMeans(n_clusters=num_centers, random_state=0)

            elif self._cluster_method == 'img_query_clust':
                if self._data_name in ['coco', 'bccd', 'pcd', 'cifar10', 'cifar100', 'svhn', 'caltech101', 'lgchem']:
                    for category in self._dataset.categories().values():
                        if isinstance(category, LabelCategories):
                            num_centers = len(list(category._indices.keys()))
                center_dict = {i: [] for i in range(num_centers)}
                temp_dataset = self._dataset
                for item in temp_dataset:
                    for anno in item.annotations:
                        if isinstance(anno, Label):
                            label_ = anno.label
                            if not center_dict.get(label_):
                                center_dict[label_] = item
                    if all(center_dict.values()):
                        break
                centroids = [self._database_keys[self._item_list.index(
                    i)] for i in list(center_dict.values())]
                kmeans = KMeans(n_clusters=num_centers, init=centroids, random_state=0)

            elif self._cluster_method == 'txt_query_clust':
                if self._data_name in ['coco', 'bccd', 'pcd', 'cifar10', 'cifar100', 'svhn', 'caltech101', 'lgchem']:
                    for category in self._dataset.categories().values():
                        if isinstance(category, LabelCategories):
                            labels = list(category._indices.keys())
                            num_centers = len(labels)
                centroids = []
                for label in labels:
                    prompt = "a photo of {}".format(label)
                    inputs = tokenize(prompt)
                    hash_key = hash_inference_text(inputs)[0]
                    hash_key = np.unpackbits(hash_key, axis=-1)
                    centroids.append(hash_key)

                kmeans = KMeans(n_clusters=num_centers,init=centroids, random_state=0)

            elif self._cluster_method in ['img_txt_query_clust', 'img_txt_prompt_query_clust', 'img_txt_coop_query_clust']:
                if self._data_name in ['coco', 'bccd', 'pcd', 'cifar10', 'cifar100', 'svhn', 'caltech101', 'lgchem']:
                    for category in self._dataset.categories().values():
                        if isinstance(category, LabelCategories):
                            num_centers = len(list(category._indices.keys()))
                center_dict = {i: [] for i in range(num_centers)}
                temp_dataset = self._dataset
                label_hash = []
                for item in temp_dataset:
                    for anno in item.annotations:
                        if isinstance(anno, Label):
                            label_ = anno.label
                            if not center_dict.get(label_):
                                center_dict[label_] = item
                            if self._cluster_method == 'img_txt_query_clust':
                                prompt = "a photo of {}".format(label_)
                            elif self._cluster_method == 'img_txt_coop_query_clust':
                                prompt = [template.format(label_) for template in coop_template]
                                prompt = (" ").join(prompt)
                            else:
                                if self._data_name == 'cifar10':
                                    prompt = [template.format(label_) for template in cifar10_templates]
                                elif self._data_name == 'cifar100':
                                    prompt = [template.format(label_) for template in cifar100_templates]
                                elif self._data_name == 'caltech101':
                                    prompt = [template.format(label_) for template in caltech101_templates]
                                elif self._data_name == 'lgchem':
                                    prompt = [template.format(label_) for template in lgchem_templates]
                                elif self._data_name == 'svhn':
                                    prompt = [template.format(label_) for template in svhn_templates]
                                prompt = (" ").join(prompt)
                            inputs = tokenize(prompt)
                            hash_key = hash_inference_text(inputs)[0]
                            hash_key = np.unpackbits(hash_key, axis=-1)
                            label_hash.append(hash_key)
                    if all(center_dict.values()):
                        break
                centroids = [self._database_keys[self._item_list.index(i)] for i in list(center_dict.values())]
                kmeans = KMeans(n_clusters=num_centers, init=centroids, random_state=0)

            clusters = kmeans.fit_predict(self._database_keys)
            cluster_centers = kmeans.cluster_centers_

            removed_items = []
            for cluster in range(num_centers):
                cluster_center = cluster_centers[cluster]
                cluster_items_idx = np.where(clusters == cluster)[0]
                if self._cluster_method == 'centroid':
                    num_selected_item = 1
                elif self._cluster_method in ['prune_close', 'clustered_random', 'img_query_clust', 'txt_query_clust',  'img_txt_coop_query_clust', 'img_txt_query_clust', 'img_txt_prompt_query_clust']:
                    num_items = len(cluster_items_idx)
                    # num_selected_item = math.ceil(num_items*self._ratio)
                    num_selected_item = int(np.round(num_items*self._ratio))

                if self._cluster_method == 'clustered_random':
                    random.shuffle(cluster_items_idx)
                    for idx in cluster_items_idx[num_selected_item:]:
                        removed_items.append(self._item_list[idx])
                else:
                    cluster_items = self._database_keys[cluster_items_idx, ]
                    dist = calculate_hamming(cluster_center, cluster_items)
                    ind = np.argsort(dist)
                    item_list = cluster_items_idx[ind]
                    for idx in item_list[num_selected_item:]:
                        removed_items.append(self._item_list[idx])

            if i == 0:
                removed_items_1 = removed_items
            elif i == 1:
                removed_items_2 = removed_items
            elif i == 2:
                removed_items_3 = removed_items
            elif i == 3:
                removed_items_4 = removed_items
            elif i == 4:
                removed_items_5 = removed_items
            elif i == 5:
                removed_items_6 = removed_items

        return removed_items_1, removed_items_2, removed_items_3, removed_items_4, removed_items_5, removed_items_6
