import logging
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from .datasets import FashionMNIST_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _data_transforms_fashion_mnist():
    """创建Fashion-MNIST数据集的转换"""
    # Fashion-MNIST是灰度图像，使用简单的标准化转换
    FASHION_MNIST_MEAN = [0.2860]
    FASHION_MNIST_STD = [0.3530]

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])

    return train_transform, valid_transform

def load_fashion_mnist_data(datadir):
    """加载完整的Fashion-MNIST数据集"""
    train_transform, test_transform = _data_transforms_fashion_mnist()

    fashion_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    fashion_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = fashion_train_ds.data, fashion_train_ds.target
    X_test, y_test = fashion_test_ds.data, fashion_test_ds.target

    return (X_train, y_train, X_test, y_test)

def partition_data(dataset, datadir, partition_method, n_nets, alpha):
    """以50%异质性程度分割Fashion-MNIST数据集"""
    X_train, y_train, X_test, y_test = load_fashion_mnist_data(datadir)
    
    # 创建训练和测试集的映射
    train_net_dataidx_map = {}
    test_net_dataidx_map = {}
    
    if partition_method == "hetero":
        # 为每个客户端分配主要类别
        n_classes = 10  # Fashion-MNIST有10个类别
        client_main_classes = {}
        
        for client_id in range(n_nets):
            # 每个客户端随机分配1-2个主要类别
            num_main_classes = min(2, random.randint(1, 3))
            # 随机选择类别
            available_classes = list(range(n_classes))
            random.shuffle(available_classes)
            client_main_classes[client_id] = available_classes[:num_main_classes]
            
        # 初始化客户端数据索引
        train_idx_batch = [[] for _ in range(n_nets)]
        test_idx_batch = [[] for _ in range(n_nets)]
        
        # 按类别分配训练数据
        for class_id in range(n_classes):
            # 找出该类别的所有训练数据
            train_idx_of_class = np.where(y_train == class_id)[0]
            np.random.shuffle(train_idx_of_class)
            
            # 计算每个客户端在该类别上的数据分配比例
            proportions = np.zeros(n_nets)
            
            # 主要类别获得50%的数据
            for client_id in range(n_nets):
                if class_id in client_main_classes[client_id]:
                    proportions[client_id] = 0.5 / len(client_main_classes[client_id])
            
            # 其余类别均分剩余50%数据
            for client_id in range(n_nets):
                if class_id not in client_main_classes[client_id]:
                    # 计算客户端的非主要类别数量
                    non_main_classes_count = n_classes - len(client_main_classes[client_id])
                    proportions[client_id] = 0.5 / non_main_classes_count
            
            # 归一化确保总和为1
            proportions = proportions / proportions.sum()
            
            # 计算每个客户端应该分配多少数据
            # 使用cumsum来计算分割点
            split_points = (np.cumsum(proportions) * len(train_idx_of_class)).astype(int)
            split_points = np.append(0, split_points)  # 添加起始点0
            
            # 分配训练数据
            for client_id in range(n_nets):
                start, end = split_points[client_id], split_points[client_id + 1]
                train_idx_batch[client_id].extend(train_idx_of_class[start:end])
            
            # 同样的逻辑处理测试数据
            test_idx_of_class = np.where(y_test == class_id)[0]
            np.random.shuffle(test_idx_of_class)
            
            # 使用相同的比例分配测试数据
            test_split_points = (np.cumsum(proportions) * len(test_idx_of_class)).astype(int)
            test_split_points = np.append(0, test_split_points)  # 添加起始点0
            
            for client_id in range(n_nets):
                start, end = test_split_points[client_id], test_split_points[client_id + 1]
                test_idx_batch[client_id].extend(test_idx_of_class[start:end])
        
        # 保存每个客户端的数据索引
        for client_id in range(n_nets):
            np.random.shuffle(train_idx_batch[client_id])
            train_net_dataidx_map[client_id] = train_idx_batch[client_id]
            
            np.random.shuffle(test_idx_batch[client_id])
            test_net_dataidx_map[client_id] = test_idx_batch[client_id]
            
            # 记录每个客户端的类别分布
            train_class_counts = {}
            for class_id in range(n_classes):
                train_class_counts[class_id] = np.sum(y_train[train_idx_batch[client_id]] == class_id)
            
            logger.info(f"客户端 {client_id} - 主要类别: {client_main_classes[client_id]}")
            logger.info(f"客户端 {client_id} - 训练集类别分布: {train_class_counts}")
            logger.info(f"客户端 {client_id} - 训练集样本数: {len(train_idx_batch[client_id])}, 测试集样本数: {len(test_idx_batch[client_id])}")
            
    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    """获取Fashion-MNIST的数据加载器"""
    transform_train, transform_test = _data_transforms_fashion_mnist()
    
    train_ds = FashionMNIST_truncated(datadir, dataidxs=dataidxs_train, train=True, 
                           transform=transform_train, download=True)
    test_ds = FashionMNIST_truncated(datadir, dataidxs=dataidxs_test, train=False, 
                          transform=transform_test, download=True)
    
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, 
                              shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, 
                             shuffle=False, drop_last=False)
    
    return train_dl, test_dl

def load_partition_data_fashion_mnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    """加载并分区Fashion-MNIST数据集 - 主函数"""
    # 获取分区数据
    X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map = \
        partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    
    # 计算训练和测试样本总数
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])
    
    # 类别数量
    class_num = len(np.unique(y_train))
    
    # 获取全局数据加载器
    train_data_global, test_data_global = get_dataloader(
        dataset, data_dir, batch_size, batch_size)
    
    # 创建本地数据加载器和记录每个客户端的数据量
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    for client_idx in range(client_number):
        train_dataidxs = train_net_dataidx_map[client_idx]
        test_dataidxs = test_net_dataidx_map[client_idx]
        
        # 记录每个客户端的数据量
        local_data_num = len(train_dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        
        # 创建本地数据加载器
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, 
            train_dataidxs, test_dataidxs)
            
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
        logger.info(f"客户端 {client_idx} - 训练样本数: {local_data_num}, "
                   f"测试样本数: {len(test_dataidxs)}")
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num