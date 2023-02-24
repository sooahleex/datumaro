import os
import datumaro as dm
import time
import copy

from datumaro.components.prune import Prune

data_name = 'svhn'
# cifar10 train
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/cifar10_train/cifar-10-batches-py", "cifar")
# cifar100
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/cifar100_train", format="cifar")
# svhn
dataset = dm.Dataset.import_from("/media/hdd2/datumaro/svhn/train_data", format="imagenet")
# caltech101
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/caltech101/train_data", format="imagenet")
# lg_chem
# dataset = dm.Dataset.import_from('/media/hdd2/datumaro/lgchem/train_data', format='imagenet')
print('dataset len : ', len(dataset))

hash_base_model = 'clip'
cluster_method = 'centroid' # random, centroid, img_query_clust, txt_query_clust, img_txt_query_clust
print('---- data type : ', cluster_method)
hash_type = 'img' # img_txt, txt, img, img_txt_prompt
print('---- hashing_type : ', hash_type)
Pruner = Prune(dataset, ratio_list=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8], cluster_method=cluster_method, data_name=data_name, hash_type=hash_type, hash_base_model=hash_base_model)
removed_items_1, removed_items_5, removed_items_10, removed_items_20, removed_items_50, removed_items_80 = Pruner.get_pruned()
print(f'exception items number : {len(Pruner._exception_items)}')

###### 0.01
dataset_1 = copy.deepcopy(dataset)
removed_items_1.extend(Pruner._exception_items)
print(f'removed items number for 0.01 : {len(removed_items_1)}')
removed_ids = []
removed_subsets = []
for item in removed_items_1:
    removed_ids.append(item.id)
    removed_subsets.append(item.subset)

n = 0
for id_, subset in zip(removed_ids, removed_subsets):
    dataset_1.remove(id_, subset)
    n += 1
print('remain dataset len for 0.01 : ', len(dataset_1))
print(f'{n} data removed')

print(f'{data_name}/{data_name}_1_{cluster_method}_prompts saved......')
dataset_1.export(f'prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_1_{cluster_method}', 'imagenet', save_images=True)


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

print(f'{data_name}/{data_name}_5_{cluster_method}_prompts saved......')
dataset_5.export(f'prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_5_{cluster_method}', 'imagenet', save_images=True)

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

print(f'{data_name}/{data_name}_10_{cluster_method}_prompts saved......')
dataset_10.export(f'prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_10_{cluster_method}', 'imagenet', save_images=True)

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

print(f'{data_name}/{data_name}_20_{cluster_method}_prompts saved......')
dataset_20.export(f'prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_20_{cluster_method}', 'imagenet', save_images=True)

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

print(f'{data_name}/{data_name}_50_{cluster_method}_prompts saved......')
dataset_50.export(f'prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_50_{cluster_method}', 'imagenet', save_images=True)

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

print(f'{data_name}/{data_name}_80_{cluster_method}_prompts saved......')
dataset.export(f'prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_80_{cluster_method}', 'imagenet', save_images=True)
