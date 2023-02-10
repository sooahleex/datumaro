import os
import datumaro as dm
import time
import copy

from datumaro.components.prune import Prune

# dataset = dm.Dataset.import_from('./tests/assets/coco_dataset/coco')
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/coco_dataset", format='coco_instances')
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/vitens_test")
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/coco_train", format='coco_instances')
# print('dataset len : ', len(dataset))

# ratio = 0.2
# for ratio in [0.05, 0.1]:
data_name = 'bccd'
dataset = dm.Dataset.import_from(f"/media/hdd2/datumaro/{data_name}_test")
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/coco_dataset", format='coco_instances')
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/coco_train", format='coco_instances')
print('dataset len : ', len(dataset))
data_type = 'query_text'

Pruner = Prune(dataset, ratio_list=[0.05, 0.1], data_type=data_type, data_name=data_name)
removed_items_5, removed_items_10 = Pruner.get_pruned()

dataset_5 = copy.deepcopy(dataset)
removed_items_5.extend(Pruner._exception_items)
removed_items_10.extend(Pruner._exception_items)
print(f'exception items number : {len(Pruner._exception_items)}')
print(f'removed items number for 0.05 : {len(removed_items_5)}')
print(f'removed items number for 0.1 : {len(removed_items_10)}')

removed_ids = []
removed_subsets = []
for item in removed_items_5:
    removed_ids.append(item.id)
    removed_subsets.append(item.subset)

n = 0
for id_, subset in zip(removed_ids, removed_subsets):
    dataset_5.remove(id_, subset)
    n += 1
print('remain dataset len for 0.05 : ', len(dataset_5))
print(f'{n} data removed')

print(f'{data_name}/pruned_{data_name}_5_{data_type} saved......')
dataset_5.export(f'{data_name}/pruned_{data_name}_5_{data_type}', 'coco_instances', save_images=True)

removed_ids = []
removed_subsets = []
for item in removed_items_10:
    removed_ids.append(item.id)
    removed_subsets.append(item.subset)

n = 0
for id_, subset in zip(removed_ids, removed_subsets):
    dataset.remove(id_, subset)
    n += 1
print('remain dataset len for 0.1 : ', len(dataset))
print(f'{n} data removed')

print(f'{data_name}/pruned_{data_name}_10_{data_type} saved......')
dataset.export(f'{data_name}/pruned_{data_name}_10_{data_type}', 'coco_instances', save_images=True)
