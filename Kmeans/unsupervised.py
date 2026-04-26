import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14          # 全局默认字体大小
plt.rcParams['axes.titlesize'] = 18     # 各个子图标题的大小
plt.rcParams['axes.labelsize'] = 16     # X轴和Y轴标签的大小
plt.rcParams['xtick.labelsize'] = 12    # X轴刻度数字的大小
plt.rcParams['ytick.labelsize'] = 12    # Y轴刻度数字的大小

# 1. 数据加载与预处理
# --------------------------------------------------
# 加载波士顿房价数据集（新版本sklearn需使用fetch_openml）
data = fetch_openml(name='boston', version=1, as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)
target = data.target

# 数据标准化（K-means和线性回归都需要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 2. 非监督特征增强
# --------------------------------------------------
# 使用K-means生成聚类标签（注意仅在训练集上训练）
kmeans = KMeans(n_clusters=3, random_state=42)

# 先划分数据集再训练聚类器（避免数据泄漏）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
kmeans.fit(X_train)  # 仅在训练集上学习聚类模式

# 为所有数据生成聚类标签
train_clusters = kmeans.predict(X_train).reshape(-1, 1)
test_clusters = kmeans.predict(X_test).reshape(-1, 1)

# 特征增强：将聚类标签作为新特征
X_train_aug = np.hstack([X_train, train_clusters])
X_test_aug = np.hstack([X_test, test_clusters])

# 3. 监督回归建模
# --------------------------------------------------
# 原始特征模型
model_orig = LinearRegression()
model_orig.fit(X_train, y_train)
y_pred_orig = model_orig.predict(X_test)

# 增强特征模型
model_aug = LinearRegression()
model_aug.fit(X_train_aug, y_train)
y_pred_aug = model_aug.predict(X_test_aug)

# 4. 结果对比
# --------------------------------------------------
# 性能指标计算
print("原始特征模型：")
print(f"MSE: {mean_squared_error(y_test, y_pred_orig):.2f}")
print(f"R²: {r2_score(y_test, y_pred_orig):.2f}\n")

print("增强特征模型：")
print(f"MSE: {mean_squared_error(y_test, y_pred_aug):.2f}")
print(f"R²: {r2_score(y_test, y_pred_aug):.2f}")

# 可视化结果对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_orig, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('原始特征模型预测结果')
plt.xlabel('真实房价')
plt.ylabel('预测房价')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_aug, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('增强特征模型预测结果')
plt.xlabel('真实房价')
plt.ylabel('预测房价')

plt.tight_layout()
plt.show()