# 优化器
class SGDOptimizer:
    def __init__(self, learning_rate=0.01, decay_rate=0.0):
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        
    def update_learning_rate(self):
        """使用学习率衰减"""
        if self.decay_rate > 0:
            self.learning_rate = self.initial_learning_rate / (1 + self.decay_rate * self.iterations)
        self.iterations += 1
        
    def get_learning_rate(self):
        return self.learning_rate