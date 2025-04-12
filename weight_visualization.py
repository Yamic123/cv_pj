import mymodel
from struct import unpack
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import load_para

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

input_size = 3072  
output_size = 10

# 加载最佳参数和模型路径
best_params, best_model_path = load_para.load_best_info()

# 创建神经网络模型结构
layer_sizes = [input_size, best_params['hidden_size'], output_size]
activations = [best_params['activation'], 'sigmoid']
model = mymodel.NeuralNetwork(layer_sizes, activations)
model.load_model(r'./cv_pj/final_model.pkl')

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    # 转换字节键名
    images = dict[b'data'].reshape(-1, 3, 32, 32)
    labels = np.array(dict[b'labels'])
    return images, labels

# 加载测试数据
test_imgs, test_labels = load_cifar10_batch(
        os.path.join('.\dataset\cifar-10-batches-py', 'test_batch'))

# 归一化
test_imgs = test_imgs / test_imgs.max()

# 获取权重矩阵
mats = []
mats.append(model.layers[0].weights)  # 输入层到隐藏层的权重 (3072×hidden_size)
mats.append(model.layers[1].weights)  # 隐藏层到输出层的权重 (hidden_size×10)

# 原始权重矩阵可视化
plt.figure(figsize=(20, 15))
plt.subplot(1, 2, 1)
plt.matshow(mats[0], fignum=0)
plt.title("输入层到隐藏层权重矩阵")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.matshow(mats[1], fignum=0)
plt.title("隐藏层到输出层权重矩阵")
plt.colorbar()

plt.tight_layout()
plt.savefig("权重矩阵原始可视化.png", dpi=300)
plt.show()

# 将输入层到隐藏层的权重转换为图像块可视化
def weights_to_img(weights, shape=(32, 32, 3)):
    """将权重向量转换为图像形状"""
    # 确保权重是正确的形状
    if weights.shape[0] != np.prod(shape):
        raise ValueError(f"权重向量大小 {weights.shape[0]} 与目标形状 {shape} 不匹配")
    
    # 重塑权重为图像形状
    img = weights.reshape(shape)
    
    # 如果是RGB图像，需要将通道维度移到最后
    if len(shape) == 3:
        # 假设形状是(channels, height, width)，转换为(height, width, channels)
        img = np.transpose(img, (1, 2, 0))
    
    # 归一化到[0,1]以便可视化
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    return img

# 可视化输入层到隐藏层的前16个神经元权重（3072->32x32x3）
plt.figure(figsize=(15, 15))
plt.suptitle("输入层到隐藏层的前16个神经元权重可视化", fontsize=8)

for i in range(16):
    if i < mats[0].shape[1]:  # 确保有足够的神经元
        # 获取第i个隐藏神经元的所有连接权重
        neuron_weights = mats[0][:, i]
        
        # 将权重转换为32x32x3的图像
        img = weights_to_img(neuron_weights, shape=(3, 32, 32))
        
        # 绘制图像
        plt.subplot(4, 4, i+1)
        plt.imshow(img)
        plt.title(f"神经元 #{i+1}")
        plt.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以容纳总标题
plt.savefig("输入层到隐藏层权重可视化.png", dpi=300)
plt.show()

# 可视化隐藏层到输出层的10个输出神经元权重
plt.figure(figsize=(15, 8))
plt.suptitle("隐藏层到输出层的10个输出神经元权重可视化", fontsize=8)

for i in range(10):
    # 获取第i个输出神经元的所有连接权重
    neuron_weights = mats[1][:, i]
    
    # 计算最接近的平方数以便形成一个正方形图像
    side_length = int(np.sqrt(len(neuron_weights)))
    
    # 如果不是完美平方数，我们可能需要裁剪或填充
    # 这里简单地取一个16x16块或填充到16x16
    target_size = min(16, side_length)
    
    # 创建一个16x16的图像(或更小，如果没有足够的权重)
    img = np.zeros((target_size, target_size))
    for j in range(min(target_size*target_size, len(neuron_weights))):
        img[j // target_size, j % target_size] = neuron_weights[j]
    
    # 归一化到[0,1]以便可视化
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # 绘制图像
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='viridis')
    plt.title(f"输出 #{i} (类别 {i})")
    plt.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("隐藏层到输出层权重可视化.png", dpi=300)
plt.show()
