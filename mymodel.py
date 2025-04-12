import numpy as np
import op
import os
import pickle

# 神经网络模型
class NeuralNetwork:
    def __init__(self, layer_sizes, activations, seed=None):
        """
        初始化神经网络
        
        参数:
            layer_sizes: 列表，指定每层的大小，包括输入层、隐藏层和输出层
            activations: 列表，指定每层的激活函数，长度应该比layer_sizes少1
            seed: 随机数种子，用于结果的再现性
        """
        if seed is not None:
            np.random.seed(seed)
            
        if len(layer_sizes) - 1 != len(activations):
            raise ValueError("激活函数的数量应该比层数少1")
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(op.Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))
        
        self.best_params = None
        self.best_val_accuracy = 0
        
    def forward(self, X):
        """前向传播"""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
            
        # 对最后一层应用softmax
        return op.Activation.softmax(A)
    
    def backward(self, y_pred, y_true, l2_lambda=0.0):
        """反向传播"""
        # 计算交叉熵损失的梯度
        dA = op.Loss.cross_entropy_derivative(y_pred, y_true)
        
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    
    def update_params(self, learning_rate, l2_lambda=0.0):
        """更新所有层的参数"""
        for layer in self.layers:
            layer.update_params(learning_rate, l2_lambda)
    
    def predict(self, X):
        """预测类别"""
        probas = self.forward(X)
        return np.argmax(probas, axis=1)
    
    def compute_accuracy(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        
        #true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == y)
    
    def compute_loss(self, X, y, l2_lambda=0.0):
        """计算损失，包括L2正则化"""
        y_pred = self.forward(X)
        cross_entropy_loss = op.Loss.cross_entropy(y_pred, y)
        
        # 添加L2正则化
        l2_loss = 0
        m = X.shape[0]
        if l2_lambda > 0:
            for layer in self.layers:
                l2_loss += np.sum(np.square(layer.weights))
            l2_loss *= (l2_lambda / (2 * m))
        
        return cross_entropy_loss + l2_loss
    
    def save_model(self, filepath):
        """保存模型参数"""
        model_params = []
        for layer in self.layers:
            layer_params = {
                'weights': layer.weights,
                'bias': layer.bias,
                'activation': layer.activation_name
            }
            model_params.append(layer_params)
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_params, f)
    
    def load_model(self, filepath):
        """加载模型参数"""
        with open(filepath, 'rb') as f:
            model_params = pickle.load(f)
        
        for i, params in enumerate(model_params):
            self.layers[i].weights = params['weights']
            self.layers[i].bias = params['bias']
            self.layers[i].activation_name = params['activation']
            self.layers[i].activation, self.layers[i].activation_derivative = op.Activation.get_activation(params['activation'])
    
    def save_best_params(self):
        """保存当前最佳参数"""
        self.best_params = []
        for layer in self.layers:
            layer_params = {
                'weights': layer.weights.copy(),
                'bias': layer.bias.copy(),
                'activation': layer.activation_name
            }
            self.best_params.append(layer_params)
    
    def load_best_params(self):
        """加载最佳参数"""
        if self.best_params is None:
            raise ValueError("没有保存的最佳参数")
        
        for i, params in enumerate(self.best_params):
            self.layers[i].weights = params['weights'].copy()
            self.layers[i].bias = params['bias'].copy()
            self.layers[i].activation_name = params['activation']
            self.layers[i].activation, self.layers[i].activation_derivative = op.Activation.get_activation(params['activation'])
