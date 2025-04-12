import numpy as np
import os
import pickle

# 数据预处理函数
def preprocess_data(X, y, num_classes):
    """
    预处理数据: 标准化特征并将标签转换为one-hot编码
    
    参数:
        X: 特征矩阵
        y: 标签向量
        num_classes: 类别数量
    
    返回:
        X_normalized: 标准化后的特征
        y_one_hot: one-hot编码的标签
    """
    # 标准化特征
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / (std + 1e-8)  # 避免除以0
    
    # 将标签转换为one-hot编码
    y_one_hot = np.zeros((y.size, num_classes))
    y_one_hot[np.arange(y.size), y] = 1
    
    return X_normalized, y_one_hot, mean, std

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    # 转换字节键名
    images = dict[b'data'].reshape(-1, 3, 32, 32)
    labels = np.array(dict[b'labels'])
    return images, labels


def load_cifar10():
    train_data = []
    train_labels = []
    
    # 加载训练集（5个批次）
    for i in range(1, 6):
        batch_file = os.path.join('.\dataset\cifar-10-batches-py', f'data_batch_{i}')
        images, labels = load_cifar10_batch(batch_file)
        train_data.append(images)
        train_labels.append(labels)
    
    # 合并训练集
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    # 加载测试集
    test_images, test_labels = load_cifar10_batch(
        os.path.join('.\dataset\cifar-10-batches-py', 'test_batch'))
    
    return (train_data, train_labels), (test_images, test_labels)


def preprocess_cifar10():
    # 加载原始数据
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()
    
    
    # 验证集分割
    val_images = train_images[:10000]
    val_labels = train_labels[:10000]
    train_images = train_images[10000:]
    train_labels = train_labels[10000:]
    mean = np.mean(train_images, axis=(0, 1, 2))
    std = np.std(train_images, axis=(0, 1, 2))
    
    # 对训练集、验证集和测试集进行标准化
    train_images = (train_images - mean) / (std+1e-8)
    val_images = (val_images - mean) / (std+1e-8)
    test_images = (test_images - mean) / (std+1e-8)
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)