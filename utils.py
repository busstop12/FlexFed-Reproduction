import torch
import torch.nn as nn
import torch.optim as optim


def test_model(model, test_loader, device):
    test_correct = 0
    total = 0
    model.to(device)
    model.eval()
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        total += len(target)

        _, predict = torch.max(output, 1)
        predict = predict.view(-1)
        test_correct += torch.sum(torch.eq(predict, target)).item()

    return 1. * test_correct / total


def train_model(model, train_loader, device, lr, momentum, weight_decay):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # 根据论文，每一次本地训练需要连续进行10次，然后将训练的结果返回中心服务器进行聚合
    for i in range(10):
        print(f'Round: {i}')
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 清空上次训练时的梯度
            optimizer.zero_grad()
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            # 反向传播
            loss.backward()
            optimizer.step()
    # 将本地模型的参数返回
    return model.state_dict()
