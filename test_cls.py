import os
import datumaro as dm
import time
import copy

from datumaro.components.prune import Prune

data_name = 'cifar10'
# dataset = dm.Dataset.import_from(f"/media/hdd2/datumaro/{data_name}_test")
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/coco_dataset", format='coco_instances')
# dataset = dm.Dataset.import_from("/media/hdd2/cifar10/cifar-10-batches-py", format="cifar")
# test
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/cifar10_train/cifar-10-test", "cifar")
# train
dataset = dm.Dataset.import_from("/media/hdd2/datumaro/cifar10_train/cifar-10-batches-py", "cifar")
print('dataset len : ', len(dataset))

data_type = 'query_label' # prune_centroid, query_text, query_label
print('---- data type : ', data_type)
hashing_type = 'use_label' # 'use_label'
print('---- hashing_type : ', hashing_type)
Pruner = Prune(dataset, ratio_list=[0.05, 0.1, 0.2, 0.5, 0.8], data_type=data_type, data_name=data_name, hashing_type=hashing_type)
removed_items_5, removed_items_10, removed_items_20, removed_items_50, removed_items_80 = Pruner.get_pruned()
print(f'exception items number : {len(Pruner._exception_items)}')

###### 0.05
dataset_5 = copy.deepcopy(dataset)
removed_items_5.extend(Pruner._exception_items)
print(f'removed items number for 0.05 : {len(removed_items_5)}')
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
dataset_5.export(f'prune_results/{data_name}/{hashing_type}/pruned_{data_name}_5_{data_type}', 'imagenet', save_images=True)
# dataset_5.export(f'{data_name}/pruned_{data_name}_5_{data_type}', 'imagenet', save_images=True)

###### 0.1
dataset_10 = copy.deepcopy(dataset)
removed_items_10.extend(Pruner._exception_items)
print(f'removed items number for 0.1 : {len(removed_items_10)}')
removed_ids = []
removed_subsets = []
for item in removed_items_10:
    removed_ids.append(item.id)
    removed_subsets.append(item.subset)

n = 0
for id_, subset in zip(removed_ids, removed_subsets):
    dataset_10.remove(id_, subset)
    n += 1
print('remain dataset len for 0.1 : ', len(dataset_10))
print(f'{n} data removed')

print(f'{data_name}/pruned_{data_name}_10_{data_type} saved......')
dataset_10.export(f'prune_results/{data_name}/{hashing_type}/pruned_{data_name}_10_{data_type}', 'imagenet', save_images=True)
# dataset_10.export(f'{data_name}/pruned_{data_name}_10_{data_type}', 'imagenet', save_images=True)

###### 0.2
dataset_20 = copy.deepcopy(dataset)
removed_items_20.extend(Pruner._exception_items)
print(f'removed items number for 0.2 : {len(removed_items_20)}')
removed_ids = []
removed_subsets = []
for item in removed_items_20:
    removed_ids.append(item.id)
    removed_subsets.append(item.subset)

n = 0
for id_, subset in zip(removed_ids, removed_subsets):
    dataset_20.remove(id_, subset)
    n += 1
print('remain dataset len for 0.2 : ', len(dataset_20))
print(f'{n} data removed')

print(f'{data_name}/pruned_{data_name}_20_{data_type} saved......')
dataset_20.export(f'prune_results/{data_name}/{hashing_type}/pruned_{data_name}_20_{data_type}', 'imagenet', save_images=True)
# dataset_20.export(f'{data_name}/pruned_{data_name}_20_{data_type}', 'imagenet', save_images=True)

###### 0.5
dataset_50 = copy.deepcopy(dataset)
removed_items_50.extend(Pruner._exception_items)
print(f'removed items number for 0.5 : {len(removed_items_50)}')
removed_ids = []
removed_subsets = []
for item in removed_items_50:
    removed_ids.append(item.id)
    removed_subsets.append(item.subset)

n = 0
for id_, subset in zip(removed_ids, removed_subsets):
    dataset_50.remove(id_, subset)
    n += 1
print('remain dataset len for 0.5 : ', len(dataset_50))
print(f'{n} data removed')

print(f'{data_name}/pruned_{data_name}_50_{data_type} saved......')
dataset_50.export(f'prune_results/{data_name}/{hashing_type}/pruned_{data_name}_50_{data_type}', 'imagenet', save_images=True)
# dataset_50.export(f'{data_name}/pruned_{data_name}_50_{data_type}', 'imagenet', save_images=True)

###### 0.8
removed_items_80.extend(Pruner._exception_items)
print(f'removed items number for 0.8 : {len(removed_items_80)}')
removed_ids = []
removed_subsets = []
for item in removed_items_80:
    removed_ids.append(item.id)
    removed_subsets.append(item.subset)

n = 0
for id_, subset in zip(removed_ids, removed_subsets):
    dataset.remove(id_, subset)
    n += 1
print('remain dataset len for 0.8 : ', len(dataset))
print(f'{n} data removed')

print(f'{data_name}/pruned_{data_name}_80_{data_type} saved......')
dataset.export(f'prune_results/{data_name}/{hashing_type}/pruned_{data_name}_80_{data_type}', 'imagenet', save_images=True)
# dataset.export(f'{data_name}/pruned_{data_name}_80_{data_type}', 'imagenet', save_images=True)
