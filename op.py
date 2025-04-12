import numpy as np

# 激活函数和它们的导数
class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))  # 防止溢出
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x):
        # 为了数值稳定性，减去每行的最大值
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def get_activation(name):
        if name == 'sigmoid':
            return Activation.sigmoid, Activation.sigmoid_derivative
        elif name == 'relu':
            return Activation.relu, Activation.relu_derivative
        elif name == 'tanh':
            return Activation.tanh, Activation.tanh_derivative
        else:
            raise ValueError(f"不支持的激活函数: {name}")
        
    # 损失函数
class Loss:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        # 添加小的常数防止log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        m = y_true.shape[0]
        y_true = np.eye(10)[y_true]
        ce_loss = -np.sum(y_true * np.log(y_pred)) / m
        return ce_loss
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        y_true = np.eye(10)[y_true]
        return y_pred - y_true
    

# 神经网络层
class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation_name = activation
        self.activation, self.activation_derivative = Activation.get_activation(activation)
        
        # 反向传播中使用的缓存
        self.input = None
        self.z = None
        self.output = None
        
        # 梯度
        self.dW = None
        self.db = None

    def forward(self, X):
        
        self.input_shape = X.shape
        if X.ndim>2:
            X = X.reshape(X.shape[0], -1)
        self.input = X
        self.z = np.dot(X, self.weights) + self.bias
        self.output = self.activation(self.z)

        return self.output
    
    def backward(self, dA):
        # dA是从下一层传来的激活值的梯度
        dZ = dA * self.activation_derivative(self.z)
        m = self.input.shape[0]
        
        self.dW = np.dot(self.input.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        
        # 计算传递给上一层的梯度
        dA_prev = np.dot(dZ, self.weights.T)
        if self.input.ndim > 2:
            dA_prev = dA_prev.reshape(self.input_shape)
            self.dW = self.dW.reshape()
        return dA_prev
    
    def update_params(self, learning_rate, l2_lambda=0.0):
        # 更新权重和偏置，包括L2正则化
        m = self.input.shape[0]
        self.weights -= learning_rate * (self.dW + (l2_lambda / m) * self.weights)
        self.bias -= learning_rate * self.db