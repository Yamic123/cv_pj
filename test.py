import train_test
import data_precessing
import mymodel as mymodel
import load_para

if __name__ == "__main__":
# 在测试集上评估最终模型
    _,_,test_data = data_precessing.preprocess_cifar10()
    test_imgs = test_data[0]
    test_labs = test_data[1]

    input_size = 3072  
    output_size = 10

    best_params, best_model_path = load_para.load_best_info()

    layer_sizes = [input_size, best_params['hidden_size'], output_size]
    activations = [best_params['activation'], 'sigmoid']
    final_model = mymodel.NeuralNetwork(layer_sizes, activations)
    test_accuracy = train_test.test_model(final_model, test_imgs, test_labs, './cv_pj/final_model.pkl')
    print(f"最终测试集准确率: {test_accuracy:.4f}")
    