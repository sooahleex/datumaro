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
    distH = 0.5 * (q - B1@B2.transpose())
    # distH = q - B1@B2.transpose()
    return distH

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

# class Solution(object):
#    def hammingDistance(self, x, y):
#       """
#       :type x: int
#       :type y: int
#       :rtype: int
#       """
#       ans = 0
#       for i in range(31,-1,-1):
#          b1= x>>i&1
#          b2 = y>>i&1
#          ans+= not(b1==b2)
#          #if not(b1==b2):
#             # print(b1,b2,i)
#       return ans
# ob1 = Solution()
# print(ob1.hammingDistance(7, 15))



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
        query_key_list = [query_key[i:i+2] for i in range(0, len(query_key), 2)]
        query_key = np.array([int(s, 16) for s in query_key_list], dtype='uint8')
        query_key = np.unpackbits(query_key, axis=-1)
        # query_key = np.reshape(query_key, (1, -1))
        # query_key_string = np.packbits(query_key, axis=-1)
        # query_key_string = list(map(lambda row: ''.join(['{:02x}'.format(r) for r in row]), [query_key_string]))

        retrieval_keys = []
        # retrieval_keys_string =[]
        # id_list = []
        item_list = []
        for datasetitem in self._dataset:
            # hash_key = int(datasetitem.hash_key[0], 36)
            hash_key = datasetitem.hash_key[0]
            hash_key_list = [hash_key[i:i+2] for i in range(0, len(hash_key), 2)]
            hash_key = np.array([int(s, 16) for s in hash_key_list], dtype='uint8')
            hash_key = np.unpackbits(hash_key, axis=-1)
            # hash_key = np.reshape(hash_key, (1, -1))
            # hash_string = np.packbits(hash_key, axis=-1)
            # hash_string = list(map(lambda row: ''.join(['{:02x}'.format(r) for r in row]), [hash_string]))
            retrieval_keys.append(hash_key)
            # retrieval_keys_string.append(hash_string)
            # id_list.append(datasetitem.id)
            item_list.append(datasetitem)
        
        retrieval_keys = np.stack(retrieval_keys, axis=0)
        # retrieval_keys_string = np.stack(retrieval_keys_string, axis=0)
        # retrieval_keys = torch.stack(retrieval_keys, dim=1).to(device = "cpu")

        # logits = 100. * query_key @ retrieval_keys

        logits = calculate_hamming(query_key, retrieval_keys)
        # pred = logits.topk(topk, 0, True, True)[1]
        ind = np.argsort(logits)
        # retrieval_keys = retrieval_keys[ind] # reorder gnd

        # tgnd = retrieval_keys[0:topk]
        
        # id_list = np.array(id_list)[ind]
        # result = id_list[:topk].tolist()

        item_list =np.array(item_list)[ind]
        result = item_list[:topk].tolist()

        return result
