import os
import datumaro as dm
import copy

from datumaro.components.prune import Prune

dataset_path_dict = {
    "cifar10": "/media/hdd2/datumaro/cifar10_train/cifar-10-batches-py",
    "cifar100": "/media/hdd2/datumaro/cifar100_train",
    "svhn": "/media/hdd2/datumaro/svhn/train_data",
    "caltech101": "/media/hdd2/datumaro/caltech101/train_data",
    "lgchem": '/media/hdd2/datumaro/lgchem/train_data',
    "food101": '/media/hdd2/datumaro/food101/train_data'
}

dataset_format_dict = {
    "cifar10": "cifar", "cifar100": "cifar", "svhn": "imagenet", "caltech101":"imagenet", "lgchem":"imagenet", "food101":"imagenet"
}

data_name = 'cifar10'
dataset = dm.Dataset.import_from(dataset_path_dict[data_name], format=dataset_format_dict[data_name])
print(f'{data_name} dataset len : ', len(dataset))

hash_base_model = 'clip' # clip, effb0_trained, effb0_init
print('---- hash_base_model : ', hash_base_model)

# random, clustered_random, centroid, 
# img_query_clust, txt_query_clust, img_txt_query_clust, img_txt_prompt_query_clust, img_txt_coop_query_clust
# query_avg_clust, query_clust
# cls_hist, entropy, center_dist_one, center_dist_multi
# cluster_method = 'clustered_random'
# print('---- cluster_method : ', cluster_method)

hash_type = 'img_txt' # img_txt, txt, img, img_txt_prompt, img_txt_coop
print('---- hashing_type : ', hash_type)

# Pruner = Prune(dataset, ratio_list=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8], cluster_method=cluster_method, data_name=data_name, hash_type=hash_type, hash_base_model=hash_base_model)
# removed_items_1, removed_items_5, removed_items_10, removed_items_20, removed_items_50, removed_items_80 = Pruner.get_pruned()
for cluster_method in ['clustered_random', 'centroid', 'query_clust', 'query_avg_clust', 'entropy', 'center_dist_one']:
    print('---- cluster_method : ', cluster_method)
    Pruner = Prune(dataset, ratio_list=[0.01], cluster_method=cluster_method, data_name=data_name, hash_type=hash_type, hash_base_model=hash_base_model)
    removed_items_1 = Pruner.get_pruned()
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

    print(f'{data_name}_{hash_base_model}_{hash_type}_1_{cluster_method} saved.....')
    dataset_1.export(f'/media/hdd2/datumaro/prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_1_{cluster_method}', 'imagenet', save_images=True)


# ###### 0.05
# dataset_5 = copy.deepcopy(dataset)
# removed_items_5.extend(Pruner._exception_items)
# print(f'removed items number for 0.05 : {len(removed_items_5)}')
# removed_ids = []
# removed_subsets = []
# for item in removed_items_5:
#     removed_ids.append(item.id)
#     removed_subsets.append(item.subset)

# n = 0
# for id_, subset in zip(removed_ids, removed_subsets):
#     dataset_5.remove(id_, subset)
#     n += 1
# print('remain dataset len for 0.05 : ', len(dataset_5))
# print(f'{n} data removed')

# print(f'{data_name}_{hash_base_model}_{hash_type}_5_{cluster_method} saved.....')
# dataset_5.export(f'/media/hdd2/datumaro/prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_5_{cluster_method}', 'imagenet', save_images=True)

# ###### 0.1
# dataset_10 = copy.deepcopy(dataset)
# removed_items_10.extend(Pruner._exception_items)
# print(f'removed items number for 0.1 : {len(removed_items_10)}')
# removed_ids = []
# removed_subsets = []
# for item in removed_items_10:
#     removed_ids.append(item.id)
#     removed_subsets.append(item.subset)

# n = 0
# for id_, subset in zip(removed_ids, removed_subsets):
#     dataset_10.remove(id_, subset)
#     n += 1
# print('remain dataset len for 0.1 : ', len(dataset_10))
# print(f'{n} data removed')

# print(f'{data_name}_{hash_base_model}_{hash_type}_10_{cluster_method} saved......')
# dataset_10.export(f'/media/hdd2/datumaro/prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_10_{cluster_method}', 'imagenet', save_images=True)

# ###### 0.2
# dataset_20 = copy.deepcopy(dataset)
# removed_items_20.extend(Pruner._exception_items)
# print(f'removed items number for 0.2 : {len(removed_items_20)}')
# removed_ids = []
# removed_subsets = []
# for item in removed_items_20:
#     removed_ids.append(item.id)
#     removed_subsets.append(item.subset)

# n = 0
# for id_, subset in zip(removed_ids, removed_subsets):
#     dataset_20.remove(id_, subset)
#     n += 1
# print('remain dataset len for 0.2 : ', len(dataset_20))
# print(f'{n} data removed')

# print(f'{data_name}_{hash_base_model}_{hash_type}_20_{cluster_method} saved......')
# dataset_20.export(f'/media/hdd2/datumaro/prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_20_{cluster_method}', 'imagenet', save_images=True)

# ###### 0.5
# dataset_50 = copy.deepcopy(dataset)
# removed_items_50.extend(Pruner._exception_items)
# print(f'removed items number for 0.5 : {len(removed_items_50)}')
# removed_ids = []
# removed_subsets = []
# for item in removed_items_50:
#     removed_ids.append(item.id)
#     removed_subsets.append(item.subset)

# n = 0
# for id_, subset in zip(removed_ids, removed_subsets):
#     dataset_50.remove(id_, subset)
#     n += 1
# print('remain dataset len for 0.5 : ', len(dataset_50))
# print(f'{n} data removed')

# print(f'{data_name}_{hash_base_model}_{hash_type}_50_{cluster_method} saved......')
# dataset_50.export(f'/media/hdd2/datumaro/prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_50_{cluster_method}', 'imagenet', save_images=True)

# ###### 0.8
# removed_items_80.extend(Pruner._exception_items)
# print(f'removed items number for 0.8 : {len(removed_items_80)}')
# removed_ids = []
# removed_subsets = []
# for item in removed_items_80:
#     removed_ids.append(item.id)
#     removed_subsets.append(item.subset)

# n = 0
# for id_, subset in zip(removed_ids, removed_subsets):
#     dataset.remove(id_, subset)
#     n += 1
# print('remain dataset len for 0.8 : ', len(dataset))
# print(f'{n} data removed')

# print(f'{data_name}_{hash_base_model}_{hash_type}_80_{cluster_method} saved......')
# dataset.export(f'/media/hdd2/datumaro/prune_results/{data_name}/{hash_base_model}/{hash_type}/{data_name}_{hash_base_model}_{hash_type}_80_{cluster_method}', 'imagenet', save_images=True)
