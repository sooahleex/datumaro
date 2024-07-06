import numpy as np

import datumaro as dm
from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.annotation import AnnotationType, Label
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT

data_path = "/media/hdd1/multiclass_food101_large"
save_path = "./multiclass_food101_pseudo_dist"
save6k_path = "./multiclass_food101_pseudo_dist_test6k"
actual_save_path = "./multiclass_food101_pseudo_dist_actual"
actual_6k_save_path = "./multiclass_food101_pseudo_dist_actual_test6k"

detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(data_path)
dm_dataset = dm.Dataset.import_from(data_path, detected_formats[0])

actual = dm.Dataset.import_from(actual_6k_save_path, "datumaro")

###################################
"""
Check the accuracy between the pseudo label and the original label
"""

# result = dm.Dataset.import_from(save6k_path, "datumaro")
# same_result = []
# for expected in result:
#     actual = find(dm_dataset, lambda x: x.id == expected.id and x.subset == expected.subset)
#     assert expected.id == actual.id
#     assert expected.subset == actual.subset
#     assert len(expected.annotations) == len(actual.annotations)
#     for e, a in zip(expected.annotations, actual.annotations):
#         assert e.type == a.type
#         same_result.append(e.label == a.label)
# # assert len(same_result) == dm_dataset.__len__()
# print('result acc : ', sum(same_result)/len(same_result))

###################################


###################################
"""
 Select 300 train and 120 val items for each label
 to match same ratio of the topk minimum distance based result
"""

# from collections import defaultdict
# import random
# vals = defaultdict(list)
# trains = defaultdict(list)
# tests = []
# for item in dm_dataset:
#     if item.subset == 'test':
#         tests.append(item)
# for item in dm_dataset:
#     if item.subset == 'train':
#         trains[item.annotations[0].label].append(item)
#     elif item.subset == 'val':
#         vals[item.annotations[0].label].append(item)

# selected_trains = defaultdict(list)
# for label, items in trains.items():
#     selected_trains[label] = random.sample(items, 300)
# selected_vals = defaultdict(list)
# for label, items in vals.items():
#     selected_vals[label] = random.sample(items, 120)

# actual = dm.Dataset.from_iterable(tests, categories=dm_dataset.categories())
# for items in selected_trains.values():
#     for item in items:
#         actual.put(item)
# for items in selected_vals.values():
#     for item in items:
#         actual.put(item)
# print("Export dataset....")

# actual.export(save_dir=actual_6k_save_path, format="datumaro", save_media=True)

####################################
"""
Top-k Minimum distance-based pseudo labeling for unlabeled dataset
"""

ids = []
for item in dm_dataset:
    if item.subset in ["train", "val"]:
        ids.append((item.id, item.subset))

dm_dataset = dm_dataset.transform("remove_annotations", ids=ids)

trainset = dm_dataset.get_subset("train")
valset = dm_dataset.get_subset("val")

label_indices = dm_dataset.categories()[AnnotationType.label]._indices
labels = list(label_indices.keys())

import random
from collections import defaultdict

tests = defaultdict(list)
for item in dm_dataset:
    if item.subset in ["test"]:
        tests[item.annotations[0].label].append(item)

selected_tests = defaultdict(list)
for label, items in tests.items():
    selected_tests[label] = random.sample(items, 180)

print("Calculate hash for trainset....")
train_explorer = Explorer(trainset)
print("Calculate hash for valset....")
val_explorer = Explorer(valset)

label_hashkeys = [
    np.unpackbits(train_explorer._get_hash_key_from_text_query(label).hash_key, axis=-1)
    for label in labels
]
label_hashkeys = np.stack(label_hashkeys, axis=0)

train_result = train_explorer.explore_topk_dist(300, label_hashkeys)
val_result = val_explorer.explore_topk_dist(120, label_hashkeys)

items = []
for label, train_items in train_result.items():
    for train_item in train_items:
        items.append(
            DatasetItem(
                id=train_item.id,
                subset=train_item.subset,
                media=train_item.media,
                annotations=[Label(label)],
            )
        )

for label, val_items in val_result.items():
    for val_item in val_items:
        items.append(
            DatasetItem(
                id=val_item.id,
                subset=val_item.subset,
                media=val_item.media,
                annotations=[Label(label)],
            )
        )

result = dm.Dataset.from_iterable(items, categories=dm_dataset.categories())
for items in selected_tests.values():
    for item in items:
        result.put(item)

print("Export dataset....")
result.export(save_dir=save_path, format="datumaro", save_media=True)

####################################
"""
Get same train and val with result and use original test set
"""

# result_items = []
# for item in dm_dataset:
#     if item.subset == "test":
#         result_items.append(item)
# for item in result:
#     if item.subset == "train":
#         result_items.append(item)
#     elif item.subset == "val":
#         result_items.append(item)

# result_6k = dm.Dataset.from_iterable(result_items, categories=dm_dataset.categories())

# print("Export dataset....")
# result_6k.export(save_dir=save6k_path, format="datumaro", save_media=True)
