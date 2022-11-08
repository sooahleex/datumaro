# Copyright (C) 2022 Intel Corporationworkbench.action.openLargeOutput
#
# SPDX-License-Identifier: MIT



import os
import datumaro as dm
import time

from datumaro.components.searcher import Searcher

# dataset = dm.Dataset.import_from('./tests/assets/widerface_dataset')
start_time = time.time()
# dataset = dm.Dataset.import_from('./tests/assets/ade20k2017_dataset', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/cifar10_dataset', format="cifar", save_hash=True)
# dataset = dm.Dataset.import_from('/media/hdd1/ade20k_val', format='common_semantic_segmentation', save_hash=True)
dataset = dm.Dataset.import_from("coco_dataset", format='coco_instances', save_hash=True)
# dataset = dm.Dataset.import_from("//media/hdd1/Datasets/mfnd")
print(f'setting dataset time for {len(dataset)} items: ', time.time()-start_time)

for item in dataset:
    print(item)

for i, item in enumerate(dataset):
    if i == 1:
        query = item

searcher = Searcher(dataset)
topk_list = searcher.search_topk(query, topk=1)
print(topk_list)
