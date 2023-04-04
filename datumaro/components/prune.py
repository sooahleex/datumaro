# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import logging as log
import math
import os
import pickle
import random
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from openvino.inference_engine import IECore
from sklearn.cluster import KMeans
from tokenizers import Tokenizer
from tqdm import tqdm

from datumaro.components.annotation import Label, LabelCategories
from datumaro.components.dataset import IDataset
from clip_prompts import (
    cifar10_templates,
    cifar100_templates,
    caltech101_templates,
    lgchem_templates,
    svhn_templates,
    food101_templates,
    eurosat_templates,
)

device = "cpu"
model_folder = "./searcher"
model = {
    "IMG": None,
    "TXT": None,
}
tokenizer_ = {"clip-vit-base-patch320": None}
coop = {"checkpoint": None}


def img_center_crop(image, size):
    """
    Crop center of image
    """

    width, height = image.shape[1], image.shape[0]
    mid_w, mid_h = int(width / 2), int(height / 2)

    crop_w = size if size < image.shape[1] else image.shape[1]
    crop_h = size if size < image.shape[0] else image.shape[0]
    mid_cw, mid_ch = int(crop_w / 2), int(crop_h / 2)

    cropped_image = image[mid_h - mid_ch : mid_h + mid_ch, mid_w - mid_cw : mid_w + mid_cw]
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

    tokenizer = tokenizer_["clip-vit-base-patch320"]
    if not tokenizer:
        tokenizer = Tokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer_["clip-vit-base-patch320"] = tokenizer
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
            result[:, n : n + j] = token_
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
    tokenizer = tokenizer_["clip-vit-base-patch320"]
    if not tokenizer:
        tokenizer = Tokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer_["clip-vit-base-patch320"] = tokenizer
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


def load_checkpoint(fpath):
    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(fpath, map_location=map_location)
    checkpoint_ = checkpoint["state_dict"]
    return checkpoint_


def coop_hash(class_hash):
    checkpoint = coop["checkpoint"]
    if not checkpoint:
        coop_path = "./coop_vit_b32_nctx16_s1/prompt_learner/model.pth.tar-50"
        checkpoint = load_checkpoint(coop_path)
        coop["checkpoint"] = checkpoint
    ctx = checkpoint["ctx"]
    text_hash = np.concatenate(
        (_compute_hash(ctx.detach().cpu().numpy()), np.expand_dims(class_hash, axis=0))
    )
    return text_hash

# random.seed(0)

class Prune:
    def __init__(
        self,
        dataset: IDataset,
        ratio_list: List[float],
        cluster_method: str,
        data_name: str,
        hash_type: str = None,
        hash_base_model: str = None,
    ) -> None:
        self._dataset = dataset
        self._ratio_list = ratio_list
        self._cluster_method = cluster_method
        self._data_name = data_name
        self._hash_type = hash_type

        database_keys = None
        item_list = []
        exception_items = []
        labels = []

        if self._data_name in [
            "coco",
            "bccd",
            "pcd",
            "cifar10",
            "cifar100",
            "svhn",
            "caltech101",
            "lgchem",
            "food101",
            "eurosat",
        ]:
            for category in self._dataset.categories().values():
                if isinstance(category, LabelCategories):
                    num_centers = len(list(category._indices.keys()))
        else:
            num_centers = len(self._dataset.subsets())
        self._num_centers = num_centers

        if self._cluster_method == "random":
            for datasetitem in self._dataset:
                item_list.append(datasetitem)
        else:
            try:
                with open(
                    f"./hash_pickles/{self._data_name}_{hash_base_model}_{self._hash_type}.pickle",
                    "rb",
                ) as handle:
                    saved_dict = pickle.load(handle)
                    database_keys = saved_dict["database_keys"]
                    labels = saved_dict["labels"]
                    item_list = saved_dict["item_list"]
            except:
                if self._hash_type == "img_txt":
                    category_dict = {}
                    for label, indice in list(self._dataset.categories().values())[
                        0
                    ]._indices.items():
                        category_dict[indice] = f"a photo of {label}"
                elif self._hash_type == "img_txt_prompt":
                    category_dict = {}
                    for label, indice in list(self._dataset.categories().values())[
                        0
                    ]._indices.items():
                        if self._data_name == "cifar10":
                            category_dict[indice] = [
                                template.format(label) for template in cifar10_templates
                            ]
                        elif self._data_name == "cifar100":
                            category_dict[indice] = [
                                template.format(label) for template in cifar100_templates
                            ]
                        elif self._data_name == "caltech101":
                            category_dict[indice] = [
                                template.format(label) for template in caltech101_templates
                            ]
                        elif self._data_name == "lgchem":
                            category_dict[indice] = [
                                template.format(label) for template in lgchem_templates
                            ]
                        elif self._data_name == "svhn":
                            category_dict[indice] = [
                                template.format(label) for template in svhn_templates
                            ]
                        elif self._data_name == "food101":
                            category_dict[indice] = [
                                template.format(label) for template in food101_templates
                            ]
                        elif self._data_name == "eurosat":
                            category_dict[indice] = [
                                template.format(label) for template in eurosat_templates
                            ]
                elif self._hash_type == "img_txt_coop":
                    category_dict = {}
                    for label, indice in list(self._dataset.categories().values())[
                        0
                    ]._indices.items():
                        category_dict[indice] = label

                for datasetitem in tqdm(self._dataset):
                    try:
                        if self._hash_type == 'img':
                            hash_key = hash_inference(dataset.media.data)[0]

                        if self._hash_type in ["img_txt", "img_txt_prompt"]:
                            hash_key_img = hash_inference(datasetitem.media.data)[0]
                            prompt = category_dict.get(datasetitem.annotations[0].label)
                            if isinstance(prompt, List):
                                prompt = (" ").join(prompt)
                            inputs = tokenize(prompt)
                            hash_key_txt = hash_inference_text(inputs)[0]
                            hash_key = np.concatenate([hash_key_img, hash_key_txt])

                        if self._hash_type in ["img_txt_coop"]:
                            hash_key_img = hash_inference(datasetitem.media.data)[0]
                            prompt = category_dict.get(datasetitem.annotations[0].label)
                            inputs = tokenize(prompt)
                            hash_key_txt = hash_inference_text(inputs)[0]
                            hash_key = np.concatenate(
                                [hash_key_img, np.reshape(coop_hash(hash_key_txt), -1)]
                            )
                        else:
                            hash_key = hash_inference(datasetitem.media.data)[0]
                        hash_key = np.unpackbits(hash_key, axis=-1)

                        if database_keys is None:
                            database_keys = hash_key.reshape(1, -1)
                        else:
                            database_keys = np.concatenate(
                                (database_keys, hash_key.reshape(1, -1)), axis=0
                            )
                        item_list.append(datasetitem)
                        labels.append(datasetitem.annotations[0].label)
                    except Exception:
                        exception_items.append(datasetitem)
                save_dict = {
                    "database_keys": database_keys,
                    "item_list": item_list,
                    "labels": labels,
                }
                with open(
                    f"./hash_pickles/{self._data_name}_{hash_base_model}_{self._hash_type}.pickle",
                    "wb",
                ) as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"{self._data_name}_{hash_base_model}_{self._hash_type}.pickle saved.......")

        self._database_keys = database_keys
        self._item_list = item_list
        self._labels = labels
        self._exception_items = exception_items

    def get_pruned(self) -> None:
        dataset_len = len(self._item_list)

        if self._cluster_method == "random":
            for r_idx, ratio in enumerate(self._ratio_list):
                self._ratio = ratio

                removed_items = []
                # num_selected_item = math.ceil(dataset_len * self._ratio)
                num_selected_item = int(101 * self._ratio * 100)
                random_list = random.sample(range(dataset_len), num_selected_item)
                removed_items = list(range(dataset_len))
                for idx in random_list:
                    removed_items.remove(idx)
                removed_items = (np.array(self._item_list)[removed_items]).tolist()

                if r_idx == 0:
                    removed_items_1 = removed_items
                elif r_idx == 1:
                    removed_items_2 = removed_items
                elif r_idx == 2:
                    removed_items_3 = removed_items
                elif r_idx == 3:
                    removed_items_4 = removed_items
                elif r_idx == 4:
                    removed_items_5 = removed_items
                elif r_idx == 5:
                    removed_items_6 = removed_items
            # return removed_items_1, removed_items_2, removed_items_3, removed_items_4, removed_items_5, removed_items_6
            return removed_items_1, removed_items_2, removed_items_3, removed_items_4

        for r_idx, ratio in enumerate(self._ratio_list):
            self._ratio = ratio

            if self._cluster_method == "centroid":
                # self._num_centers = math.ceil(len(self._database_keys) * self._ratio)
                self._num_centers = int(101 * self._ratio * 100)
                # kmeans = KMeans(n_clusters=self._num_centers, random_state=0)
                kmeans = KMeans(n_clusters=self._num_centers)

            elif self._cluster_method in [
                "prune_close",
                "clustered_random",
                "cls_hist",
                "entropy",
                "center_dist_one",
                "center_dist_multi",
            ]:
                # kmeans = KMeans(n_clusters=self._num_centers, random_state=0)
                kmeans = KMeans(n_clusters=self._num_centers)

            elif self._cluster_method == "query_clust":
                center_dict = {i: [] for i in range(self._num_centers)}
                temp_dataset = self._dataset
                for item in temp_dataset:
                    for anno in item.annotations:
                        if isinstance(anno, Label):
                            label_ = anno.label
                            if not center_dict.get(label_):
                                center_dict[label_] = item
                    if all(center_dict.values()):
                        break
                # centroids = [
                #     self._database_keys[self._item_list.index(i)]
                #     for i in list(center_dict.values())
                # ]
                centroids = []
                item_id_list = [item.id for item in self._item_list]
                for idx, i in enumerate(list(center_dict.values())):
                    centroids.append(self._database_keys[item_id_list.index(i.id)])
                # kmeans = KMeans(n_clusters=self._num_centers, n_init=1, init=centroids, random_state=0)
                kmeans = KMeans(n_clusters=self._num_centers, n_init=1, init=centroids)


            elif self._cluster_method in ["query_txt_clust", "query_txt_clust_center_dist_one", "query_txt_clust_entropy"]:
                if self._data_name in [
                    "coco",
                    "bccd",
                    "pcd",
                    "cifar10",
                    "cifar100",
                    "svhn",
                    "caltech101",
                    "lgchem",
                    "food101",
                    "eurosat",
                ]:
                    for category in self._dataset.categories().values():
                        if isinstance(category, LabelCategories):
                            labels = list(category._indices.keys())
                centroids = []
                for label in labels:
                    prompt = "a photo of a {}".format(label)
                    inputs = tokenize(prompt)
                    hash_key = hash_inference_text(inputs)[0]
                    hash_key = np.unpackbits(hash_key, axis=-1)
                    centroids.append(hash_key)

                # kmeans = KMeans(n_clusters=self._num_centers, n_init=1, init=centroids, random_state=0)
                kmeans = KMeans(n_clusters=self._num_centers, n_init=1, init=centroids)

            elif self._cluster_method == "query_avg_clust":
                center_dict = {i: [] for i in range(self._num_centers)}
                items_per_label_dict = {i: None for i in range(self._num_centers)}
                temp_dataset = self._dataset
                item_id_list = [item.id for item in self._item_list]
                for item in temp_dataset:
                    for anno in item.annotations:
                        if isinstance(anno, Label):
                            label_ = anno.label
                            if items_per_label_dict.get(label_) is None:
                                items_per_label_dict[label_] = self._database_keys[
                                    item_id_list.index(item.id)
                                ]
                            else:
                                items_per_label_dict[label_] = np.vstack(
                                    (
                                        items_per_label_dict.get(label_),
                                        self._database_keys[item_id_list.index(item.id)],
                                    )
                                )
                for label, hashes in items_per_label_dict.items():
                    center_key = np.mean(hashes, axis=0)
                    center_dict[label] = center_key
                # kmeans = KMeans(
                #     n_clusters=self._num_centers,
                #     n_init=1,
                #     init=list(center_dict.values()),
                #     random_state=0,
                # )
                kmeans = KMeans(n_clusters=self._num_centers, n_init=1, init=list(center_dict.values()))

            clusters = kmeans.fit_predict(self._database_keys)
            cluster_centers = kmeans.cluster_centers_
            cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)

            ######### match num item for each cluster with labels num
            if self._cluster_method != 'centroid':
                total_num_selected_item = int(self._num_centers * self._ratio * 100)
                norm_cluster_num_item_list = [int(self._ratio*100) for _ in range(self._num_centers)]
                added_num = 0
                for idx_clust, n_clust in enumerate(cluster_num_item_list):
                    if n_clust < int(self._ratio*100):
                        num_gap = int(self._ratio*100) - n_clust
                        norm_cluster_num_item_list[idx_clust] = n_clust
                        added_num += num_gap
                sorted_cluster_num_item_list = np.argsort(cluster_num_item_list)[::-1]
                for idx_clust in sorted_cluster_num_item_list[:added_num]:
                    norm_cluster_num_item_list[idx_clust] += 1
                    assert norm_cluster_num_item_list[idx_clust] <= cluster_num_item_list[idx_clust]

            ######### match num item for each cluster
            # total_num_selected_item = math.ceil(dataset_len * self._ratio)

            # cluster_num_item_list = [
            #     float(i) / sum(cluster_num_item_list) * total_num_selected_item
            #     for i in cluster_num_item_list
            # ]
            # norm_cluster_num_item_list = [int(np.round(i)) for i in cluster_num_item_list]
            # zero_cluster_indexes = list(np.where(np.array(norm_cluster_num_item_list) == 0)[0])
            # add_clust_dist = np.sort(np.array(cluster_num_item_list)[zero_cluster_indexes])[::-1][
            #     : total_num_selected_item - sum(norm_cluster_num_item_list),
            # ]
            # for dist in set(add_clust_dist):
            #     indices = [i for i, x in enumerate(cluster_num_item_list) if x == dist]
            #     for index in indices:
            #         norm_cluster_num_item_list[index] += 1
            # if total_num_selected_item > sum(norm_cluster_num_item_list):
            #     diff_num_item_list = np.argsort(
            #         np.array(
            #             [
            #                 x - norm_cluster_num_item_list[i]
            #                 for i, x in enumerate(cluster_num_item_list)
            #             ]
            #         )
            #     )[::-1]
            #     for diff_idx in diff_num_item_list[
            #         : total_num_selected_item - sum(norm_cluster_num_item_list)
            #     ]:
            #         norm_cluster_num_item_list[diff_idx] += 1
            # elif total_num_selected_item < sum(norm_cluster_num_item_list):
            #     diff_num_item_list = np.argsort(
            #         np.array(
            #             [
            #                 x - norm_cluster_num_item_list[i]
            #                 for i, x in enumerate(cluster_num_item_list)
            #             ]
            #         )
            #     )
            #     for diff_idx in diff_num_item_list[
            #         : sum(norm_cluster_num_item_list) - total_num_selected_item
            #     ]:
            #         norm_cluster_num_item_list[diff_idx] -= 1

            removed_items = []
            selected_item_indexs = []
            for cluster_id in cluster_ids:
                cluster_center = cluster_centers[cluster_id]
                cluster_items_idx = np.where(clusters == cluster_id)[0]
                if self._cluster_method == "centroid":
                    num_selected_item = 1
                else:
                    num_selected_item = norm_cluster_num_item_list[cluster_id]

                if self._cluster_method == "clustered_random":
                    random.shuffle(cluster_items_idx)
                    for idx in cluster_items_idx[num_selected_item:]:
                        removed_items.append(self._item_list[idx])
                elif self._cluster_method == "cls_hist":
                    cluster_items = self._database_keys[
                        cluster_items_idx,
                    ]
                    dist = np.linalg.norm(cluster_center - cluster_items, axis=-1)
                    ind = np.argsort(dist)
                    clustered_item_list = cluster_items_idx[ind]
                    n, _, _ = plt.hist(dist)
                    sum_n = 0
                    max_index = None
                    if sum([int(np.round(i * ratio)) for i in n]) == 0:
                        max_index = n.argmax()
                    for j, n_ in enumerate((n)):
                        n_ = int(n_)
                        n_list = clustered_item_list[
                            sum_n : sum_n + n_,
                        ]
                        num_selected_item = int(np.round(n_ * ratio))
                        if max_index and j == max_index:
                            num_selected_item = 1
                        selected_items = random.sample(n_list.tolist(), num_selected_item)
                        sum_n += n_
                        for selected_item in selected_items:
                            selected_item_indexs.append(selected_item)
                elif self._cluster_method in ["entropy", "query_txt_clust_entropy"]:
                    cltr_classes = np.array(self._labels)[cluster_items_idx]
                    _, inv, cnts = np.unique(
                        cltr_classes, return_counts=True, return_inverse=True
                    )

                    weights = 1 / cnts
                    probs = weights[inv]
                    probs = probs / probs.sum()

                    choices = np.random.choice(
                        range(len(inv)), size=num_selected_item, p=probs, replace=False
                    )
                    assert len(choices) == num_selected_item
                    assert len(np.unique(choices)) == len(choices)
                    selected_item_indexs += [cluster_items_idx[choices]]
                elif self._cluster_method in ["center_dist_one", 'query_txt_clust_center_dist_one']:
                    c_dist = calculate_hamming(cluster_center, cluster_centers)
                    near_c = cluster_centers[np.argsort(c_dist)[1]]

                    cluster_items = self._database_keys[
                        cluster_items_idx,
                    ]
                    dist_btw_near_c = calculate_hamming(near_c, cluster_items)
                    ind_btw_near_c = np.argsort(dist_btw_near_c)
                    item_indices_btw_near_c = cluster_items_idx[ind_btw_near_c]

                    dist = calculate_hamming(cluster_center, cluster_items)
                    ind = np.argsort(dist)
                    item_indices = cluster_items_idx[ind]
                    if num_selected_item > 1:
                        # select nearest one for center of cluster
                        # selected_item_indexs.append(item_indices[0])
                        # select nearest items for center of close cluster
                        selected_item_indexs.extend(item_indices_btw_near_c[:num_selected_item])
                    elif num_selected_item == 1:
                        # select nearest one for center of cluster
                        selected_item_indexs.append(item_indices[0])

                elif self._cluster_method == "center_dist_multi":
                    selected_item_indexs_for_c = []
                    c_dist_indices = np.argsort(calculate_hamming(cluster_center, cluster_centers))
                    cluster_items = self._database_keys[
                        cluster_items_idx,
                    ]

                    dist = calculate_hamming(cluster_center, cluster_items)
                    ind = np.argsort(dist)
                    item_indices = cluster_items_idx[ind]

                    near_item_for_clust_dict = {}
                    for c_dist_indice in c_dist_indices[1:]:
                        near_c = cluster_centers[c_dist_indice]
                        dist_btw_near_c = calculate_hamming(near_c, cluster_items)
                        ind_btw_near_c = np.argsort(dist_btw_near_c)
                        item_indices_btw_near_c = cluster_items_idx[ind_btw_near_c]
                        near_item_for_clust_dict[c_dist_indice] = item_indices_btw_near_c

                    item_idx = 0
                    if num_selected_item > 0:
                        while len(selected_item_indexs_for_c) > num_selected_item:
                            for i, (c_indice, item_indices_btw_near_c) in enumerate(
                                near_item_for_clust_dict.items()
                            ):
                                if num_selected_item == 1:
                                    # select nearest one for center of cluster
                                    selected_item_indexs_for_c.append(item_indices[0])
                                    break
                                else:
                                    # select nearest one for center of cluster
                                    if i == 0 and item_idx == 0:
                                        selected_item_indexs_for_c.append(item_indices[0])
                                    elif (
                                        item_indices_btw_near_c[item_idx]
                                        not in selected_item_indexs_for_c
                                    ):
                                        selected_item_indexs_for_c.append(
                                            item_indices_btw_near_c[item_idx]
                                        )
                                        if c_indice == len(cluster_centers):
                                            item_idx += 1

                    selected_item_indexs.extend(selected_item_indexs_for_c)
                else:
                    cluster_items = self._database_keys[
                        cluster_items_idx,
                    ]
                    dist = calculate_hamming(cluster_center, cluster_items)
                    ind = np.argsort(dist)
                    item_list = cluster_items_idx[ind]
                    for idx in item_list[num_selected_item:]:
                        removed_items.append(self._item_list[idx])

            if self._cluster_method in ["cls_hist", "center_dist_one", "center_dist_multi", "query_txt_clust_center_dist_one"]:
                dataset_len = len(self._item_list)
                removed_items_index = list(range(dataset_len))
                for idx in selected_item_indexs:
                    removed_items_index.remove(idx)
                for idx in removed_items_index:
                    removed_items.append(self._item_list[idx])
            elif self._cluster_method in ["entropy", "query_txt_clust_entropy"]:
                selected_item_indexs = np.concatenate(selected_item_indexs)
                assert len(selected_item_indexs) == len(np.unique(selected_item_indexs))
                np.random.shuffle(selected_item_indexs)
                selected_item_indexs = selected_item_indexs[:total_num_selected_item]

                dataset_len = len(self._item_list)
                removed_items_index = list(range(dataset_len))
                for idx in selected_item_indexs:
                    removed_items_index.remove(idx)
                for idx in removed_items_index:
                    removed_items.append(self._item_list[idx])

            if r_idx == 0:
                removed_items_1 = removed_items
            elif r_idx == 1:
                removed_items_2 = removed_items
            elif r_idx == 2:
                removed_items_3 = removed_items
            elif r_idx == 3:
                removed_items_4 = removed_items
            elif r_idx == 4:
                removed_items_5 = removed_items
            elif r_idx == 5:
                removed_items_6 = removed_items

        # return removed_items_1, removed_items_2, removed_items_3, removed_items_4, removed_items_5, removed_items_6
        return removed_items_1, removed_items_2, removed_items_3, removed_items_4
        # return removed_items_1
