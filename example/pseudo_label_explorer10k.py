import random
import time

import numpy as np

import datumaro as dm
from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.annotation import AnnotationType, Label
from datumaro.components.environment import DEFAULT_ENVIRONMENT

start_time = time.time()
data_path = "/media/hdd1/multiclass_food101_large"
save_path = "./multiclass_food101_pseudo_high_quality_train1k"

detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(data_path)
dm_dataset = dm.Dataset.import_from(data_path, detected_formats[0])


###################################
"""
Check the accuracy between the pseudo label and the original label
"""
from datumaro.util import find

result = dm.Dataset.import_from(save_path, "datumaro")

same_result = []
for expected in result:
    if expected.subset in ["train", "val"]:
        actual = find(dm_dataset, lambda x: x.id == expected.id and x.subset == expected.subset)
        assert expected.id == actual.id
        assert expected.subset == actual.subset
        # assert len(expected.annotations) == len(actual.annotations)
        if len(expected.annotations) > 0:
            for e, a in zip(expected.annotations, actual.annotations):
                assert e.type == a.type
                same_result.append(e.label == a.label)
assert len(same_result) == 1400
print("result acc : ", sum(same_result) / len(same_result))

###################################

trainset = dm_dataset.get_subset("train")
valset = dm_dataset.get_subset("val")
train_explorer = Explorer(trainset)
val_explorer = Explorer(valset)

label_indices = dm_dataset.categories()[AnnotationType.label]._indices
labels = list(label_indices.keys())
label_hashkeys = [
    np.unpackbits(train_explorer._get_hash_key_from_text_query(label).hash_key, axis=-1)
    for label in labels
]
label_hashkeys = np.stack(label_hashkeys, axis=0)

train_result = train_explorer.explore_topk_dist(50, label_hashkeys)
val_result = val_explorer.explore_topk_dist(20, label_hashkeys)
final_items = []
for label, train_items in train_result.items():
    for train_item in train_items:
        train_item.annotations = [Label(label)]
        final_items.append(train_item)
for label, val_items in val_result.items():
    for val_item in val_items:
        val_item.annotations = [Label(label)]
        final_items.append(val_item)

rest_train_items = []
rest_val_items = []
for item in trainset:
    if item not in final_items:
        rest_train_items.append(item)
for item in valset:
    if item not in final_items:
        rest_val_items.append(item)

selected_unlabeld_train_items = random.sample(rest_train_items, 7500)
for item in selected_unlabeld_train_items:
    item.annotations = []
    final_items.append(item)
selected_unlabeld_val_items = random.sample(rest_val_items, 3000)
for item in selected_unlabeld_val_items:
    item.annotations = []
    final_items.append(item)

for item in dm_dataset:
    if item.subset in ["test"]:
        final_items.append(item)

result = dm.Dataset.from_iterable(final_items, categories=dm_dataset.categories())

print("Elapsed time: ", time.time() - start_time)
print("Export dataset....")
result.export(save_dir=save_path, format="datumaro", save_media=True)
