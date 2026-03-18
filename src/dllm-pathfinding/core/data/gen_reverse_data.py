import numpy as np
import random
import os

def star_graph(degSource, pathLen, numNodes, reverse=False):
    """
    生成一个星形图推理任务：
    1. 确定起点 source 和终点 goal。
    2. 生成一条长度为 pathLen 的最短路径。
    3. 在起点处添加 degSource-1 条干扰路径。
    """
    source = np.random.randint(0, numNodes, 1)[0]
    goal = np.random.randint(0, numNodes, 1)[0]
    while goal == source:
        goal = np.random.randint(0, numNodes, 1)[0]

    path = [source]
    edge_list = []

    # 随机选择路径上的中间节点
    for _ in range(pathLen - 2):
        node = np.random.randint(0, numNodes, 1)[0]
        while node in path or node == goal:
            node = np.random.randint(0, numNodes, 1)[0]
        path.append(node)

    path.append(goal)
    # 连接最短路径
    for i in range(len(path) - 1):
        edge_list.append([path[i], path[i + 1]])

    # 添加干扰分支（从起点出发，但不通往终点）
    i = 0
    deg_nodes = set()
    while i < degSource - 1:
        node = source
        next_node = np.random.randint(0, numNodes, 1)[0]
        l = 1
        while l < pathLen:
            if next_node not in deg_nodes and next_node not in path:
                edge_list.append([node, next_node])
                deg_nodes.add(next_node)
                node = next_node
                l += 1
            next_node = np.random.randint(0, numNodes, 1)[0]
        i += 1

    random.shuffle(edge_list)
    
    # 如果 reverse 为 True，则反转目标路径
    if reverse:
        path = path[::-1]

    return path, edge_list, source, goal

def generate_and_save(n_train, n_test, degSource, pathLen, numNodes, reverse=False):
    """
    生成并保存数据集
    """
    # 设置输出目录
    out_dir = './data/datasets/graphs/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    suffix = '_rev' if reverse else ''
    train_file = os.path.join(out_dir, f'deg_{degSource}_path_{pathLen}_nodes_{numNodes}_train_{n_train}{suffix}.txt')
    test_file = os.path.join(out_dir, f'deg_{degSource}_path_{pathLen}_nodes_{numNodes}_test_{n_test}{suffix}.txt')

    print(f">>> 开始生成训练集 (reverse={reverse})")
    print(f">>> 目标文件: {train_file}")
    with open(train_file, 'w') as f:
        for i in range(n_train):
            path, edge_list, start, goal = star_graph(degSource, pathLen, numNodes, reverse=reverse)
            path_str = ','.join(map(str, path))
            edge_str = '|'.join([f"{e[0]},{e[1]}" for e in edge_list])
            # 格式：边列表/起点,终点=路径
            f.write(f"{edge_str}/{start},{goal}={path_str}\n")
            if (i + 1) % 100000 == 0:
                print(f"进度: 已完成 {i + 1} / {n_train}")

    print(f"\n>>> 开始生成测试集")
    print(f">>> 目标文件: {test_file}")
    with open(test_file, 'w') as f:
        for i in range(n_test):
            path, edge_list, start, goal = star_graph(degSource, pathLen, numNodes, reverse=reverse)
            path_str = ','.join(map(str, path))
            edge_str = '|'.join([f"{e[0]},{e[1]}" for e in edge_list])
            f.write(f"{edge_str}/{start},{goal}={path_str}\n")

    print("\n[成功] 数据集生成完毕！")

if __name__ == "__main__":
    # 配置参数
    N_TRAIN = 800000
    N_TEST = 80000
    DEG = 5
    PATH_LEN = 12
    NUM_NODES = 100
    REVERSE = True # 开启反转
    
    generate_and_save(
        n_train=N_TRAIN, 
        n_test=N_TEST, 
        degSource=DEG, 
        pathLen=PATH_LEN, 
        numNodes=NUM_NODES, 
        reverse=REVERSE
    )
