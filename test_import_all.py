import datumaro as dm

data_name='svhn'
# dataset = dm.Dataset.import_from("/media/hdd2/datumaro/cifar10_train/cifar-10-batches-py", "cifar")
dataset = dm.Dataset.import_from("/media/hdd2/datumaro/svhn/train_data", format="imagenet")

print('dataset len : ', len(dataset))

dataset.export(f'prune_results/{data_name}/{data_name}_100','imagenet', save_images=True)
