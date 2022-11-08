# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional

import numpy as np

import torch

from datumaro.components.dataset import IDataset
from datumaro.components.extractor import DatasetItem


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    # distH = 0.5 * (q - B1@B2.transpose())
    distH = q - B1@B2.transpose()
    return distH

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))


class Searcher:
    def __init__(
        self,
        dataset: IDataset,
        topk: int = 10,
    ) -> None:
        """
        Searcher for Datumaro dataitems

        Parameters
        ----------
        dataset:
            Datumaro dataset to search similar dataitem.
        topK:

        """
        self._dataset = dataset
        self._topk = topk

    def search_topk(self, item: DatasetItem, topk: Optional[int]=None):
        """
        Search topk similar results based on hamming distance for query DatasetItem
        """
        if not topk:
            topk = self._topk

        # query_key = np.array([int(item.hash_key[0], 36)])
        query_key = item.hash_key[0]

        retrieval_keys = []
        id_list = []
        for datasetitem in self._dataset:
            # hash_key = int(datasetitem.hash_key[0], 36)
            hash_key = datasetitem.hash_key[0]
            retrieval_keys.append(hash_key)
            id_list.append(datasetitem.id)
        
        retrieval_keys = np.stack(retrieval_keys, axis=0)
        # retrieval_keys = torch.stack(retrieval_keys, dim=1).to(device = "cpu")

        # logits = 100. * query_key @ retrieval_keys

        logits = calculate_hamming(query_key, retrieval_keys)
        # pred = logits.topk(topk, 0, True, True)[1]
        ind = np.argsort(logits)
        # retrieval_keys = retrieval_keys[ind] # reorder gnd

        # tgnd = retrieval_keys[0:topk]
        id_list = np.array(id_list)[ind]
        result = id_list[:topk].tolist()

        return result
