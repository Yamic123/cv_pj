import  data_precessing
import mymodel
import op
import train_test
import myoptimizer as myoptimizer
import plot
import load_para

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
    
    best_params, best_model_path = load_para.load_best_info()

    # 使用最佳参数训练最终模型
    layer_sizes = [input_size, best_params['hidden_size'], output_size]
    activations = [best_params['activation'], 'sigmoid']
    final_model = mymodel.NeuralNetwork(layer_sizes, activations)
    if best_model_path is not None:
        final_model.load_model(best_model_path)
    
    
    history = train_test.train_model(
        final_model, train_imgs, train_labs, valid_imgs, valid_labs,
        epochs=100, learning_rate=best_params['learning_rate'],
        lr_decay=best_params['lr_decay'], l2_lambda=best_params['l2_lambda'],
        model_save_path='final_model.pkl', verbose=True
    )
    
    # 可视化训练过程
    plot.plot_training_history(history)
    
    