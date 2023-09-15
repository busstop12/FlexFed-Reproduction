import copy
import torch


def standalone(client_models):
    # 每个用户单独训练
    return client_models


def basic_common(client_models):
    # 用于聚合各个client训练的结果
    # 根据伪代码A.2
    common_list = get_common_layers(client_models)

    for k in common_list:
        # 计算共同参数的平均值
        common_weight = copy.deepcopy(client_models[0][k])
        for i in range(1, len(client_models)):
            common_weight += client_models[i][k]
        common_weight = common_weight / len(client_models)

        # 聚合共同参数
        for i in range(0, len(client_models)):
            client_models[i][k] = common_weight
    return client_models


def clustered_common(client_models):
    # 根据伪代码A.3
    # 先执行basic-common策略
    client_models_basic = basic_common(client_models)

    # 将在同一个组中的模型进行聚合
    client_num = len(client_models)
    client_per_group = client_num // 4
    for i in range(4):
        weight_copy = copy.deepcopy(client_models_basic[i * client_per_group])
        # 求取这个组中每个参数的平均值
        for key in weight_copy.keys():
            if 'classifier' in key:
                break
            for j in range(1, client_per_group):
                weight_copy[key] += client_models_basic[i * client_per_group + j][key]
            weight_copy[key] = torch.div(weight_copy[key], client_per_group)

        for j in range(client_per_group):
            client_models_basic[i * client_per_group + j] = copy.deepcopy(weight_copy)

    return client_models_basic


def max_common(client_models):
    # 根据算法A.1
    # 具体实施的时候，采取C(2, n)的方式进行选取，将两个模型进行比较与加和
    # 同时，维护一个和每个模型列表相同大小的二维数组，初始化为1，用于记录求取平均值时所用的除数

    # 所有模型的名字数组
    weight_names = []
    for i in range(len(client_models)):
        weight_names.append([s for s in client_models[i].keys()])

    # 生成除数数组
    divider = []
    for i in range(len(client_models)):
        divider.append([1 for _ in range(len(weight_names[i]))])

    # 首先将模型copy一份，因为一会做选取的时候直接在原来模型的基础上进行操作
    models_copy = copy.deepcopy(client_models)

    # 抽取两个模型，比较他们相同之处，并进行聚合
    for i in range(len(client_models)):
        for j in range(i + 1, len(client_models)):
            for k in range(len(weight_names[i])):
                name = weight_names[i][k]
                if 'classifier' in name:
                    # 最后的全连接层不进行聚合，直接结束
                    print(name)
                    break
                elif weight_names[j][k] == name:
                    # 如果当前层是一样的，进行聚合，并将除数数组加1，方便后续求取平均值
                    client_models[i][name] += models_copy[j][name]
                    client_models[j][name] += models_copy[i][name]
                    divider[i][k] += 1
                    divider[j][k] += 1
                else:
                    # 遇到第一个不等于的参数时，便可以结束
                    break

    # 求取模型参数的平均值
    for i in range(len(client_models)):
        for j in range(len(weight_names[i])):
            client_models[i][weight_names[i][j]] = client_models[i][weight_names[i][j]].cpu() / divider[i][j]

    return client_models


def get_common_layers(client_models):
    common_list = []
    min_layer = 0xffffffff
    for i in range(0, len(client_models)):
        print(len(client_models[i]))
        if len(client_models[i]) < min_layer:
            min_layer = len(client_models[i])

    # 所有模型的名字数组
    weight_names = []
    for i in range(len(client_models)):
        weight_names.append([s for s in client_models[i].keys()])

    for i in range(min_layer):
        temp = weight_names[0][i]
        flag = False
        for j in range(1, len(client_models)):
            if temp != weight_names[j][i]:
                flag = True
                break
        if flag:
            break
        else:
            common_list.append(temp)

    return common_list
