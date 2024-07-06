import numpy as np

import datumaro as dm
from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.annotation import AnnotationType
from datumaro.components.environment import DEFAULT_ENVIRONMENT

data_path = "/media/hdd1/multiclass_food101_large"
save_path = "./multiclass_food101_pseudo"
detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(data_path)
dm_dataset = dm.Dataset.import_from(data_path, detected_formats[0])

###################################
"""
Check the accuracy between the pseudo label and the original label
"""
# from datumaro.util import find
# result = dm.Dataset.import_from(save_path, "datumaro")

# same_result = []
# for expected in result:
#     actual = find(dm_dataset, lambda x: x.id == expected.id and x.subset == expected.subset)
#     assert expected.id == actual.id
#     assert expected.subset == actual.subset
#     assert len(expected.annotations) == len(actual.annotations)
#     for e, a in zip(expected.annotations, actual.annotations):
#         assert e.type == a.type
#         same_result.append(e.label == a.label)
# assert len(same_result) == dm_dataset.__len__()
# print("result acc : ", sum(same_result) / len(same_result))

###################################
"""
Pseudo labeling via Minimum Distance Assignment for Unlabeled Datasets
"""

dataset = dm.Dataset.from_iterable([])
explorer = Explorer(dataset)

label_indices = dm_dataset.categories()[AnnotationType.label]._indices
labels = list(label_indices.keys())
label_hashkeys = [
    np.unpackbits(explorer._get_hash_key_from_text_query(label).hash_key, axis=-1)
    for label in labels
]
label_hashkeys = np.stack(label_hashkeys, axis=0)

ids = []
for item in dm_dataset:
    if item.subset in ["train", "val"]:
        ids.append((item.id, item.subset))

dm_dataset = dm_dataset.transform("remove_annotations", ids=ids)

# for item in dm_dataset:
#     hashkey_ = np.unpackbits(explorer._get_hash_key_from_item_query(resultitem).hash_key)
#     logits = calculate_hamming(hashkey_, label_hashkeys)
#     ind = np.argsort(logits)
#     pseudo = np.array(labels)[ind][0]
#     pseudo_annotation = [Label(label=label_indices[pseudo])]
#     item = item.wrap(annotations=pseudo_annotation)

result = dm_dataset.transform(
    "update_annotations", label_hashkeys=label_hashkeys, explorer=explorer
)
result.export(save_dir=save_path, format="datumaro", save_media=True)
