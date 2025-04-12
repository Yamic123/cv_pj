import os
import json

def load_best_info():
    """ 从保存的json文件中加载最佳参数和模型路径 """
    file_path = os.path.join("./cv_pj/best_models", "best_info.json")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        best_params = data['best_params']
        best_model_path = data['best_model_path']
        
        # 检查路径是否存在（可选）
        if not os.path.exists(best_model_path):
            print(f"警告: 模型文件 {best_model_path} 不存在")
            
        return best_params, best_model_path
    
    except FileNotFoundError:
        print(f"错误: 未找到文件 {file_path}，请先运行超参数搜索")
        return None, None
    except json.JSONDecodeError:
        print(f"错误: {file_path} 文件格式损坏")
        return None, None