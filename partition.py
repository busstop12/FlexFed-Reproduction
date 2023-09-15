import torch
import torch.utils.data
from torchvision import transforms, datasets
import numpy as np


def split_iid(dataset, num_clients):
    dataset_length = len(dataset)

    indices = np.arange(dataset_length)
    np.random.shuffle(indices)
    split = np.array_split(indices, num_clients)

    subsets = [torch.utils.data.Subset(dataset, indices=subset_indices) for subset_indices in split]
    return subsets


def get_cifar_10(num_clients):
    # cifar-10数据集三个通道的平均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)

    client_train = split_iid(train_dataset, num_clients)
    client_test = split_iid(test_dataset, num_clients)

    return train_dataset, test_dataset, client_train, client_test


def get_dataset(dataset_name, num_clients):
    if dataset_name == 'CIFAR-10':
        return get_cifar_10(num_clients)
