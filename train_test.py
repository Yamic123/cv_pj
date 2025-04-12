import numpy as np
import myoptimizer


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=32, learning_rate=0.01, 
                lr_decay=0.0, l2_lambda=0.0, early_stopping_patience=5,
                model_save_path='best_model.pkl', verbose=True):
    """
    训练神经网络模型
    
    参数:
        model: NeuralNetwork实例
        X_train, y_train: 训练数据和标签
        X_val, y_val: 验证数据和标签
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 初始学习率
        lr_decay: 学习率衰减率
        l2_lambda: L2正则化强度
        early_stopping_patience: 早停耐心值
        model_save_path: 保存最佳模型的路径
        verbose: 是否打印训练进度
    
    返回:
        history: 包含训练和验证损失及准确率的字典
    """
    optimizer = myoptimizer.SGDOptimizer(learning_rate, lr_decay)
    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_accuracy = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # 打乱训练数据
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_train_loss = 0
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # 前向传播
            y_pred = model.forward(X_batch)
            
            # 反向传播
            model.backward(y_pred, y_batch, l2_lambda)
            
            # 更新参数
            optimizer.update_learning_rate()
            current_lr = optimizer.get_learning_rate()
            model.update_params(current_lr, l2_lambda)
            
            # 计算批次损失
            batch_loss = model.compute_loss(X_batch, y_batch, l2_lambda)
            epoch_train_loss += batch_loss * (end_idx - start_idx) / n_samples
        
        # 计算训练集和验证集的损失和准确率
        train_accuracy = model.compute_accuracy(X_train, y_train)
        val_loss = model.compute_loss(X_val, y_val, l2_lambda)
        val_accuracy = model.compute_accuracy(X_val, y_val)
        
        # 保存历史记录
        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_best_params()
            model.save_model(model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"早停: 验证集准确率连续{early_stopping_patience}个epoch没有提升")
            break
        
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}, 学习率: {current_lr:.6f}, "
                  f"训练损失: {epoch_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}, "
                  f"验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
    
    # 恢复最佳模型参数
    model.load_best_params()
    
    return history

# 测试函数
def test_model(model, X_test, y_test, model_path=None):
    """
    测试神经网络模型
    
    参数:
        model: NeuralNetwork实例
        X_test, y_test: 测试数据和标签
        model_path: 模型路径，如果提供则会先加载模型
    
    返回:
        accuracy: 测试集准确率
    """
    if model_path is not None:
        model.load_model(model_path)
        
    test_accuracy = model.compute_accuracy(X_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    return test_accuracy