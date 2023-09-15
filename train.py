import torch.utils.data
import copy
import csv

import model.VGG as VGG
import partition
import utils
import strategy


def train(config, device):
    epochs = 50
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    num_clients = 8
    batch_size = 64
    num_per_group = num_clients / 4
    model_type = 'VGG'
    dataset_name = 'CIFAR-10'
    strategy_name = config

    # 加载数据集
    train_dataset, test_dataset, client_train, client_test = partition.get_dataset(dataset_name, num_clients)

    # 装入DataLoader
    train_loaders = [
        torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
        for train_data in client_train
    ]
    test_loaders = [
        torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        for test_data in client_test
    ]

    # 全局测试集loader
    global_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    print(len(test_dataset))
    # 客户本地的模型，初始化为空
    client_model = {index: None for index in range(num_clients)}

    # 开始执行FlexiFed的不同策略
    print(f'Start training: \nModel: {model_type}\nDataset: {dataset_name}\nStrategy: {strategy_name}')

    for epoch in range(epochs):
        acc_clients = [epoch]
        for index in range(num_clients):
            if index < num_per_group:
                model = VGG.get_vgg_model('VGG11', batch_norm=True)
            elif num_per_group <= index < 2 * num_per_group:
                model = VGG.get_vgg_model('VGG13', batch_norm=True)
            elif 2 * num_per_group <= index < 3 * num_per_group:
                model = VGG.get_vgg_model('VGG16', batch_norm=True)
            else:
                model = VGG.get_vgg_model('VGG19', batch_norm=True)
            if epoch > 0:
                model.load_state_dict(client_model[index], strict=False)

            global_acc = utils.test_model(model, global_test_loader, device)
            acc_clients.append(global_acc)
            acc = utils.test_model(model, test_loaders[index], device)
            acc_clients.append(acc)
            print(f'Epoch: {epoch}, Client: {index}, Local Accuracy: {acc}, Global Accuracy: {global_acc}')

            model_copy = copy.deepcopy(model)
            local_model = utils.train_model(model_copy, train_loaders[index], device, lr, momentum, weight_decay)
            client_model[index] = copy.deepcopy(local_model)

        with open(f'{model_type}_{dataset_name}_{strategy_name}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(acc_clients)

        # 执行不同的模型聚合策略
        if strategy_name == 'STANDALONE':
            client_model = strategy.standalone(client_model)
        elif strategy_name == 'BASIC-COMMON':
            client_model = strategy.basic_common(client_model)
        elif strategy_name == 'CLUSTERED-COMMON':
            client_model = strategy.clustered_common(client_model)
        elif strategy_name == 'MAX-COMMON':
            client_model = strategy.max_common(client_model)
