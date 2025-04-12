# CV Project - Neural Network Training & Analysis Framework
# 模型架构预览
 ![Model Structure Preview](https://github.com/user-attachments/assets/228c3490-6931-4408-bfa4-af8771d7d91d)
# 核心代码结构预览
![Main_Code_Structure_Preview](https://github.com/user-attachments/assets/ba1470fd-a855-494c-b5e7-6838fd1b4518)


## 项目结构
### CV-PJ

- **best_models/**  # 最佳模型存储
  - best_info.json  # 最佳超参数配置记录

- **saved_models/**  # 超参数搜索过程保存的所有模型

- **pycache/**  # 缓存文件夹

- **data_processing.py**  # 数据预处理模块
- **hyperparameter_analysis.py**  # 超参数影响可视化
- **hyperparameter_search.py**  # 超参数网格搜索
- **load_para.py**  # 最佳参数/模型加载器
- **main.py**  # 主训练程序
- **test.py**  # 模型测试程序
- **train_test.py**  # 训练/测试核心函数
- **final_model.pkl**  # 最终部署模型

- **mymodel.py**  # 模型架构定义
- **myoptimizer.py**  # 自定义优化器
- **op.py**  # 模型组件库
- **neural_network_params.py**  # 神经网络参数管理

- **plot.py**  # 训练曲线绘制工具
- **weight_visualization.py**  # 权重可视化工具

- **training_history.png**  # 自动生成的训练曲线图
- **输入层到隐藏层权重可视化.png**  # 权重矩阵可视化结果
- **隐藏层到输出层权重可视化.png**
- **...其他可视化图片**

## 快速开始

### 环境配置
```bash
pip install numpy matplotlib seaborn
```
### 训练模型
```bash
# 基础训练（使用默认参数）
python main.py

# 超参数搜索模式
python hyperparameter_search.py

# 生成训练曲线图（自动保存为 training_history.png）
python plot.py
```
### 测试模型
```bash
# 测试最终模型
python test.py --model final_model.pkl

# 加载最佳参数测试
python test.py --config best_models/best_info.json
```
## 关键功能说明
### 数据流处理
```bash
from data_processing import DataPipeline

pipeline = DataPipeline(normalize=True, augment=True)
train_loader, test_loader = pipeline.load_data(batch_size=64)
```
### 模型架构
![image](https://github.com/user-attachments/assets/e02b6f23-2787-401a-a92b-a2ab72b7a529)
架构定义见 mymodel.py

### 权重可视化
```bash
# 生成权重矩阵图（PNG格式）
python weight_visualization.py \
    --model final_model.pkl \
    --layer_names input_hidden hidden_output \
    --save_dir ./
```
## 结果展示

### 训练曲线
![image](https://github.com/user-attachments/assets/23c5f3df-7406-41e3-8973-37fea07c04f3)

### 权重展示
![image](https://github.com/user-attachments/assets/b62ae3b1-34f9-4dcc-bccd-827321ead2ea)
![image](https://github.com/user-attachments/assets/41b70e81-48cb-4e08-98d5-df79e41f3e1e)
![image](https://github.com/user-attachments/assets/52e2ae76-647f-4226-b901-c4fe4439c10f)

## 扩展功能
### 超参数分析
```bash
python hyperparameter_analysis.py --log_dir saved_models/
```

## 贡献指南
模型修改请更新 mymodel.py 和 op.py

新增可视化功能请继承 plot.py 基类

重大参数调整需通过 hyperparameter_search.py 验证

