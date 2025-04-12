import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import neural_network_params


sns.set_theme(style='white',font_scale=0.7,font='simhei')
# 创建数据框
data = neural_network_params.data
df = pd.DataFrame(data)

# 设置图表风格
plt.figure(figsize=(30, 20))

# 1. 隐藏层大小与准确率的关系
plt.subplot(2, 2, 1)
plt.title("准确率与隐藏层大小的关系")
sns.barplot(x="隐藏层大小", y="value", hue="variable", 
            data=pd.melt(df, id_vars=["隐藏层大小"], 
                         value_vars=["验证准确率", "测试准确率"]))
plt.ylabel("准确率")
plt.xlabel("隐藏层大小",fontsize=10)
plt.legend(title="准确率类别", fontsize=10, title_fontsize=12)

# 2. L2正则化与准确率的关系
plt.subplot(2, 2, 2)
plt.title("准确率与L2正则化的关系")
sns.barplot(x="L2正则化", y="value", hue="variable", 
            data=pd.melt(df, id_vars=["L2正则化"], 
                         value_vars=["验证准确率", "测试准确率"]))
plt.ylabel("准确率")
plt.xlabel("L2正则化",fontsize=10)
plt.legend(title="准确率类型", fontsize=10, title_fontsize=12)

plt.savefig("神经网络超参数可视化1.png", dpi=300)
plt.show()

plt.figure(figsize=(30,25))
# 3. 学习率与准确率的关系
plt.subplot(3, 2, 3)
plt.title("准确率与学习率的关系")
sns.barplot(x="学习率", y="value", hue="variable", 
            data=pd.melt(df, id_vars=["学习率"], 
                         value_vars=["验证准确率", "测试准确率"]))
plt.ylabel("准确率")
plt.xlabel("学习率",fontsize=10)
plt.legend(title="准确率类型", fontsize=10, title_fontsize=12)

# 4. 学习率衰减与准确率的关系
plt.subplot(3, 2, 4)
plt.title("准确率与学习率衰减的关系")
sns.barplot(x="学习率衰减", y="value", hue="variable", 
            data=pd.melt(df, id_vars=["学习率衰减"], 
                         value_vars=["验证准确率", "测试准确率"]))
plt.ylabel("准确率")
plt.xlabel("学习率衰减",fontsize=10)
plt.legend(title="准确率类型", fontsize=10, title_fontsize=12)

plt.savefig("神经网络超参数可视化2.png", dpi=300)
plt.show()

plt.figure(figsize=(30,25))
# 5. 激活函数与准确率的关系
plt.subplot(3, 2, 5)
plt.title("准确率与激活函数的关系")
sns.barplot(x="激活函数", y="value", hue="variable", 
            data=pd.melt(df, id_vars=["激活函数"], 
                         value_vars=["验证准确率", "测试准确率"]))
plt.ylabel("准确率")
plt.xlabel("激活函数",fontsize=10)
plt.legend(title="准确率类型", fontsize=10, title_fontsize=12)

# 6. 参数组合与测试准确率的散点图
plt.subplot(3, 2, 6)
plt.title("参数组合与测试准确率的关系")

# 为激活函数创建颜色映射
activation_types = df["激活函数"].unique()
colors = {"relu": "red", "tanh": "blue", "sigmoid": "green"}

# 使用大小表示学习率，颜色表示激活函数，不透明度表示学习率衰减
for activation in activation_types:
    subset = df[df["激活函数"] == activation]
    # 将学习率映射到点的大小
    sizes = subset["学习率"].map({0.001: 50, 0.01: 100, 0.1: 200})
    # 将学习率衰减映射到透明度
    alphas = subset["学习率衰减"].map({0.0: 0.5, 0.01: 0.7, 0.1: 0.9})
    
    for i, row in subset.iterrows():
        plt.scatter(row["隐藏层大小"], row["L2正则化"], 
                   s=sizes.iloc[0], alpha=alphas.iloc[0], 
                   color=colors[activation], 
                   label=f"{activation}" if i == subset.index[0] else "")
        
        # 为每个点添加测试准确率标签
        plt.annotate(f"{row['测试准确率']:.4f}", 
                    (row["隐藏层大小"], row["L2正则化"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

plt.xlabel("隐藏层大小")
plt.ylabel("L2正则化", fontsize=12)

# 添加图例
plt.legend(title="激活函数", loc="upper right")

# 添加说明文本
plt.figtext(0.5, 0.01, "说明：点的大小表示学习率(大=0.1, 中=0.01, 小=0.001)，\n"
           "透明度表示学习率衰减(深=0.1, 中=0.01, 浅=0.0)", 
           ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("神经网络超参数可视化3.png", dpi=300)
plt.show()

# 输出参数比较表格
print("\n参数比较表:")
print(df[["隐藏层大小", "L2正则化", "学习率", "学习率衰减", "激活函数", "验证准确率", "测试准确率"]])

# 找出验证和测试准确率最高的配置
best_val = df.loc[df["验证准确率"].idxmax()]
best_test = df.loc[df["测试准确率"].idxmax()]

print("\n验证准确率最高的配置:")
print(best_val[["隐藏层大小", "L2正则化", "学习率", "学习率衰减", "激活函数", "验证准确率", "测试准确率"]])

print("\n测试准确率最高的配置:")
print(best_test[["隐藏层大小", "L2正则化", "学习率", "学习率衰减", "激活函数", "验证准确率", "测试准确率"]])

# 增加各参数的交互关系分析
plt.figure(figsize=(15, 10))
plt.suptitle("超参数交互关系分析", fontsize=16)

# 1. 隐藏层大小和激活函数的交互
plt.subplot(2, 2, 1)
plt.title("隐藏层大小和激活函数对测试准确率的影响", fontsize=12)
for act in df["激活函数"].unique():
    subset = df[df["激活函数"] == act]
    plt.plot(subset["隐藏层大小"], subset["测试准确率"], 'o-', label=f"激活函数={act}")
plt.xlabel("隐藏层大小", fontsize=10)
plt.ylabel("测试准确率", fontsize=10)
plt.legend(fontsize=8)

# 2. 学习率和L2正则化的交互
plt.subplot(2, 2, 2)
plt.title("学习率和L2正则化对测试准确率的影响", fontsize=12)
for lr in df["学习率"].unique():
    subset = df[df["学习率"] == lr]
    plt.plot(subset["L2正则化"], subset["测试准确率"], 'o-', label=f"学习率={lr}")
plt.xlabel("L2正则化", fontsize=10)
plt.ylabel("测试准确率", fontsize=10)
plt.legend(fontsize=8)

# 3. 学习率衰减和激活函数的交互
plt.subplot(2, 2, 3)
plt.title("学习率衰减和激活函数对测试准确率的影响", fontsize=12)
for act in df["激活函数"].unique():
    data = []
    for decay in sorted(df["学习率衰减"].unique()):
        subset = df[(df["激活函数"] == act) & (df["学习率衰减"] == decay)]
        if not subset.empty:
            data.append(subset["测试准确率"].mean())
    plt.plot(sorted(df["学习率衰减"].unique())[:len(data)], data, 'o-', label=f"激活函数={act}")
plt.xlabel("学习率衰减", fontsize=10)
plt.ylabel("平均测试准确率", fontsize=10)
plt.legend(fontsize=8)

# 4. 热力图展示参数间的相关性
plt.subplot(2, 2, 4)
plt.title("参数之间的相关性", fontsize=12)
# 将分类变量转换为数值
df_corr = df.copy()
df_corr["激活函数"] = df_corr["激活函数"].map({"relu": 0, "tanh": 1, "sigmoid": 2})
# 计算相关性矩阵
corr_matrix = df_corr[["隐藏层大小", "L2正则化", "学习率", "学习率衰减", "激活函数", "验证准确率", "测试准确率"]].corr()
# 绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("神经网络超参数交互分析.png", dpi=300)
plt.show()