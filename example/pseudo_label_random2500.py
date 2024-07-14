import random
import time

import numpy as np

import datumaro as dm
from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.annotation import AnnotationType, Label
from datumaro.components.environment import DEFAULT_ENVIRONMENT

start_time = time.time()
data_path = "/media/hdd1/multiclass_food101_large"
save_path = "./multiclass_food101_pseudo_random_sampling_train1k"

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

train_items = []
train_ids = []
val_items = []
val_ids = []
for item in dm_dataset:
    if item.subset in ["train"]:
        train_items.append(item)
        train_ids.append((item.id, item.subset))
    elif item.subset in ["val"]:
        val_items.append(item)
        val_ids.append((item.id, item.subset))

selected_trains_items = random.sample(train_items, 2500)
selected_trains_ids = [(item.id, item.subset) for item in selected_trains_items]
selected_vals_items = random.sample(val_items, 1000)
selected_vals_ids = [(item.id, item.subset) for item in selected_vals_items]

selected_trainset = dm.Dataset.from_iterable(
    selected_trains_items, categories=dm_dataset.categories()
)
selected_valset = dm.Dataset.from_iterable(selected_vals_items, categories=dm_dataset.categories())

train_explorer = Explorer(selected_trainset)
val_explorer = Explorer(selected_valset)

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

for item in dm_dataset:
    if item.subset == "train" and (item.id, item.subset) not in selected_trains_ids:
        item.annotations = []
        final_items.append(item)
    elif item.subset == "val" and (item.id, item.subset) not in selected_vals_ids:
        item.annotations = []
        final_items.append(item)
    elif item.subset in ["test"]:
        final_items.append(item)

result = dm.Dataset.from_iterable(final_items, categories=dm_dataset.categories())
print("Elapsed time: ", time.time() - start_time)

print("Export dataset....")
result.export(save_dir=save_path, format="datumaro", save_media=True)
