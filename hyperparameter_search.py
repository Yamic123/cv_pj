import op
import mymodel
import train_test
import os
import json
import data_precessing
import myoptimizer
# 超参数搜索


def search(X_train, y_train, X_val, y_val, X_test, y_test, input_size, output_size, params_grid, epochs=50):
    """
    执行网格搜索以找到最佳超参数
    
    参数:
        X_train, y_train: 训练数据和标签
        X_val, y_val: 验证数据和标签
        X_test, y_test: 测试数据和标签
        input_size: 输入特征数量
        output_size: 输出类别数量
        params_grid: 包含超参数列表的字典
        epochs: 每个模型的训练轮数
    
    返回:
        best_params: 最佳超参数组合
        best_model_path: 最佳模型路径
    """
    results = []
    best_val_accuracy = 0
    best_params = {}
    best_model_path = None
        
    # 这只是一个基本示例，实际的超参数搜索通常使用交叉验证并更加复杂
    for hidden_size in params_grid['hidden_size']:
        for learning_rate in params_grid['learning_rate']:
            for l2_lambda in params_grid['l2_lambda']:
                for lr_decay in params_grid['lr_decay']:
                    for activation in params_grid['activation']:
                        print(f"\n测试超参数: 隐藏层大小={hidden_size}, 学习率={learning_rate}, "
                            f"L2正则化={l2_lambda}, 学习率衰减={lr_decay}, 激活函数={activation}")
                            
                        # 创建模型架构
                        layer_sizes = [input_size, hidden_size, output_size]
                        activations = [activation, 'sigmoid']  # 最后一层使用sigmoid，然后应用softmax
                            
                        model = mymodel.NeuralNetwork(layer_sizes, activations)
                            
                            # 模型保存路径
                        model_filename = f"model_h{hidden_size}_lr{learning_rate}_l2{l2_lambda}_decay{lr_decay}_{activation}.pkl"
                        model_path = os.path.join("./cv_pj/saved_models", model_filename)
                            # 训练模型
                        history = train_test.train_model(
                            model, X_train, y_train, X_val, y_val,
                            epochs=epochs, learning_rate=learning_rate,
                            lr_decay=lr_decay, l2_lambda=l2_lambda,
                            model_save_path=model_path, verbose=False
                        )
                            
                            # 获取最佳验证准确率
                        val_accuracy = max(history['val_accuracy'])
                            
                            # 如果是最佳模型，加载并在测试集上评估
                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            best_params = {
                                'hidden_size': hidden_size,
                                'learning_rate': learning_rate,
                                'l2_lambda': l2_lambda,
                                'lr_decay': lr_decay,
                                'activation': activation
                            }
                            best_model_path = os.path.join("./cv_pj/best_models", model_filename)
                            
                            # 在测试集上评估
                        test_accuracy = train_test.test_model(model, X_test, y_test)
                            
                        results.append({
                            'hidden_size': hidden_size,
                            'learning_rate': learning_rate,
                            'l2_lambda': l2_lambda,
                            'lr_decay': lr_decay,
                            'activation': activation,
                            'val_accuracy': val_accuracy,
                            'test_accuracy': test_accuracy
                        })
                            
                        print(f"验证集准确率: {val_accuracy:.4f}, 测试集准确率: {test_accuracy:.4f}")
        
        # 打印所有结果
    print("\n超参数搜索结果:")
    for result in sorted(results, key=lambda x: x['val_accuracy'], reverse=True):
        print(f"隐藏层大小: {result['hidden_size']}, 学习率: {result['learning_rate']}, "
            f"L2正则化: {result['l2_lambda']}, 学习率衰减: {result['lr_decay']}, "
            f"激活函数: {result['activation']}, 验证准确率: {result['val_accuracy']:.4f}, "
            f"测试准确率: {result['test_accuracy']:.4f}")
        
    # 打印最佳参数
    print(f"\n最佳参数: {best_params}")
    print(f"\n最佳模型路径: {best_model_path}")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")

    best_info = {
        "best_params": best_params,
        "best_model_path": best_model_path
    }
    with open(os.path.join("./cv_pj/best_models", "best_info.json"), "w") as f:
        json.dump(best_info, f, indent=4)
        
    print(f"\n最佳参数和模型路径已保存至: {os.path.abspath('best_models')}")
        
    return best_params, best_model_path

if __name__ == "__main__":
    train_data,vali_data,test_data = data_precessing.preprocess_cifar10()
    train_imgs = train_data[0]
    train_labs = train_data[1]
    valid_imgs = vali_data[0]
    valid_labs = vali_data[1]
    test_imgs = test_data[0]
    test_labs = test_data[1]
    
     # 设置网络参数
    input_size = 3072  
    output_size = 10
    
    # 超参数搜索
    params_grid = {
        'hidden_size': [64, 128, 256],
        'learning_rate': [0.01, 0.001],
        'l2_lambda': [0, 0.001, 0.01],
        'lr_decay': [0, 0.01],
        'activation': ['relu', 'sigmoid', 'tanh']
    }
    
    best_params, best_model_path = search(
        X_train = train_imgs,
        y_train = train_labs,
        X_val = valid_imgs,
        y_val = valid_labs,
        X_test = test_imgs,
        y_test = test_labs,
        input_size = input_size,
        output_size = output_size,
        params_grid = params_grid,
        epochs=10
    )
