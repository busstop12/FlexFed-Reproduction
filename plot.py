import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_correctness_rate(file_path: str, title: str, label, rounds: int = None):
    """
    Plots the correctness rate for the specified clients.

    Parameters:
    file_path (str): The file path to the Excel file containing the correctness rate data.
    title (str): The title of the plot.
    rounds (int): The number of initial rounds to include in the plot. If None, all rounds are included.

    Returns:
    None
    """

    # 定义一组具有更多区别的颜色
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # 读取数据
    data = pd.read_excel(file_path, index_col=0)

    # 如果指定了轮数，只使用前“rounds”轮的数据
    if rounds is not None:
        data = data.iloc[:rounds]

    # 设置图表样式
    plt.figure(figsize=(10, 6))

    # 只为编号为偶数的客户端绘制线条
    for i, client in enumerate(data.columns):
        if int(client) % 2 == 0:
            plt.plot(data.index, data[client], label=label[i // 2], color=distinct_colors[i])

    # 添加图表元素
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # 设置纵轴范围为0到1
    plt.grid(True)
    plt.legend()

    # 创建存储图表的目录（如果它不存在）
    os.makedirs('graph', exist_ok=True)

    # 保存图表
    plt.savefig(f'graph/{title}.png')

    # 显示图表
    plt.show()


communicate_round = 25
client_label = [f'Client{i}' for i in range(4)]
plot_correctness_rate('VGG_CIFAR-10_STANDALONE.xlsx', 'CIFAR-10 VGG Standalone', client_label, communicate_round)
plot_correctness_rate('VGG_CIFAR-10_BASIC-COMMON.xlsx', 'CIFAR-10 VGG Basic Common', client_label, communicate_round)
plot_correctness_rate('VGG_CIFAR-10_CLUSTERED-COMMON.xlsx', 'CIFAR-10 VGG Clustered Common', client_label, communicate_round)
plot_correctness_rate('VGG_CIFAR-10_MAX-COMMON.xlsx', 'CIFAR-10 VGG Max Common', client_label, communicate_round)

plot_correctness_rate(
    'Compare.xlsx',
    'CIFAR-10 VGG',
    ['STANDALONE', 'BASIC-COMMON', 'CLUSTERED-COMMON', 'MAX-COMMON'],
    communicate_round
)
