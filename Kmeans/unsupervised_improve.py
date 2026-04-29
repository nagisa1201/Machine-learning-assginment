import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import io
import os
import warnings

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline

# 忽略不必要的警告，保持终端输出整洁
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# =====================================================================
# 消融实验开关
# =====================================================================
# 聚类参数探索分析
ENABLE_K_EXPLORATION    = True   # 开关：是否画出肘部法则、轮廓系数图，对比随机/K-means++初始化
# 特征工程选择
# FEATURE_MODE 可选值: 
# 'RAW'      : 原始标签拼接 [0, 1, 2]
# 'ONE_HOT'  : 独热编码拼接 [1,0,0], [0,1,0]
# 'DISTANCE' : 距离特征拼接 (获取样本到所有质心的距离)
# 'MULTI_K'  : 集成多K值标签 (同时拼接 K=3 和 K=5 的标签)
FEATURE_MODE            = 'ONE_HOT' 
ENABLE_INTERACTION      = False   # 开关：是否加入 房间数(RM)×聚类标签 的交互特征
# 模型选择
USE_RANDOM_FOREST       = True   # 开关：False=线性回归, True=随机森林回归器
# 高级评估与可视化
ENABLE_CROSS_VAL        = True    # 开关：是否执行5折交叉验证
ENABLE_ERROR_HIST       = True    # 开关：是否绘制预测误差分布直方图
# 工程化进阶
RUN_PIPELINE_GRIDSEARCH = False   # 开关：是否独立运行自动化管道调参 (网格搜索)
# =====================================================================

# 创建保存结果的目录
save_dir = os.path.join('result', 'experiment_improve')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"当前实验配置: 特征=[{FEATURE_MODE}], 交互=[{ENABLE_INTERACTION}], 模型=[{'RF' if USE_RANDOM_FOREST else 'LR'}]")
print("-" * 50)

# --- 数据加载 ---
data = fetch_openml(name='boston', version=1, as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)
target = data.target
# 记录 RM (房间数) 的列索引，用于后续做交互特征
rm_index = list(df.columns).index('RM')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)


# =====================================================================
# [模块一] 聚类参数优化与探索 (探索最佳K值与初始化)
# =====================================================================
if ENABLE_K_EXPLORATION:
    print("\n>>> 正在执行聚类参数优化分析...")
    k_range = [2, 3, 4, 5, 6]
    inertias, sil_scores = [], []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train)
        inertias.append(km.inertia_)
        score = silhouette_score(X_train, km.labels_)
        sil_scores.append(score)
        print(f"  K={k} | 轮廓系数={score:.4f} | 簇内误差(SSE)={km.inertia_:.0f}")
        
    # 对比初始化
    km_rand = KMeans(n_clusters=3, init='random', n_init=20, random_state=42).fit(X_train)
    km_plus = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=42).fit(X_train)
    print(f"  [初始化对比] Random_SSE: {km_rand.inertia_:.0f} vs K-means++_SSE: {km_plus.inertia_:.0f}")

    # 画图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker='o')
    plt.title('肘部法则 (Elbow Method)')
    plt.subplot(1, 2, 2)
    plt.plot(k_range, sil_scores, marker='s', color='orange')
    plt.title('轮廓系数 (Silhouette Score)')
    plt.savefig(os.path.join(save_dir, 'clustering_exploration.png'))
    plt.close()

# =====================================================================
# [模块二] 核心特征工程 (消融开关生效处)
# =====================================================================
# 基础 K=3 聚类器
base_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
base_kmeans.fit(X_train)
train_labels = base_kmeans.predict(X_train).reshape(-1, 1)
test_labels = base_kmeans.predict(X_test).reshape(-1, 1)

if FEATURE_MODE == 'RAW':
    train_feat, test_feat = train_labels, test_labels

elif FEATURE_MODE == 'ONE_HOT':
    # encoder = OneHotEncoder(sparse_output=False)
    encoder = OneHotEncoder(sparse=False)
    train_feat = encoder.fit_transform(train_labels)
    test_feat = encoder.transform(test_labels)

elif FEATURE_MODE == 'DISTANCE':
    # 直接获取样本到 3 个质心的距离 (3维特征)
    train_feat = base_kmeans.transform(X_train)
    test_feat = base_kmeans.transform(X_test)

elif FEATURE_MODE == 'MULTI_K':
    # 老同时拼接不同粒度的聚类结果
    km5 = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_train)
    train_labels_5 = km5.predict(X_train).reshape(-1, 1)
    test_labels_5 = km5.predict(X_test).reshape(-1, 1)
    train_feat = np.hstack([train_labels, train_labels_5])
    test_feat = np.hstack([test_labels, test_labels_5])

# 执行拼接
X_train_aug = np.hstack([X_train, train_feat])
X_test_aug = np.hstack([X_test, test_feat])

# 交互特征叠加
if ENABLE_INTERACTION:
    # 取出 RM (房间数) 这一列，与聚类标签相乘
    train_interact = (X_train[:, rm_index] * train_labels.flatten()).reshape(-1, 1)
    test_interact = (X_test[:, rm_index] * test_labels.flatten()).reshape(-1, 1)
    X_train_aug = np.hstack([X_train_aug, train_interact])
    X_test_aug = np.hstack([X_test_aug, test_interact])


# =====================================================================
# [模块三 & 四] 模型训练与交叉验证评估
# =====================================================================
model_aug = RandomForestRegressor(random_state=42) if USE_RANDOM_FOREST else LinearRegression()

if ENABLE_CROSS_VAL:
    # 5 折交叉验证
    scores = cross_val_score(model_aug, X_train_aug, y_train, cv=5, scoring='r2')
    print(f"\n[交叉验证] 5折 CV R² 分数: {scores}")
    print(f"[交叉验证] 平均 CV R²: {scores.mean():.4f} (标准差: {scores.std():.4f})")

# 最终测试集验证
model_aug.fit(X_train_aug, y_train)
y_pred_aug = model_aug.predict(X_test_aug)

print(f"\n[最终测试集结果]")
print(f"MSE: {mean_squared_error(y_test, y_pred_aug):.4f}")
print(f"R² : {r2_score(y_test, y_pred_aug):.4f}")


# 绘制可视化对比图与误差分布图
fig, axes = plt.subplots(1, 2 if ENABLE_ERROR_HIST else 1, figsize=(12 if ENABLE_ERROR_HIST else 6, 5))

# 散点图
ax1 = axes[0] if ENABLE_ERROR_HIST else axes
ax1.scatter(y_test, y_pred_aug, alpha=0.6, color='teal')
ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
ax1.set_title(f'预测结果 ({FEATURE_MODE} + {"RF" if USE_RANDOM_FOREST else "LR"})')
ax1.set_xlabel('真实房价')
ax1.set_ylabel('预测房价')

# 误差分布直方图
if ENABLE_ERROR_HIST:
    errors = y_test - y_pred_aug
    axes[1].hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_title('预测误差分布直方图 (Error Distribution)')
    axes[1].set_xlabel('预测误差 (真实值 - 预测值)')
    axes[1].set_ylabel('频数')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'result_{FEATURE_MODE}_int={ENABLE_INTERACTION}_RF={USE_RANDOM_FOREST}.png'), dpi=300)
plt.close()


# =====================================================================
# 进阶自动化搜索 (Pipeline + GridSearchCV)
# =====================================================================
if RUN_PIPELINE_GRIDSEARCH:
    print("\n" + "="*50)
    print(">>> 正在启动自动化管道网格搜索 (GridSearchCV)...")
    
    # 老师给的经典 Pipeline 架构
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(random_state=42)),
        ('regressor', LinearRegression())
    ])
    
    # 定义搜索空间 (聚类找最佳簇数，回归找是否需要截距)
    param_grid = {
        'kmeans__n_clusters': [2, 3, 4, 5, 6],
        'regressor__fit_intercept': [True, False]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
    
    # 注意：GridSearch 直接喂入原始 DataFrame (df)，因为 scaler 包含在 pipeline 里面了
    grid_search.fit(df, target)
    
    print("最佳参数组合:", grid_search.best_params_)
    print(f"最佳 CV R² 得分: {grid_search.best_score_:.4f}")
    print("="*50)

print(f"\n✅ 实验运行完毕！相关图表已保存至 '{save_dir}' 文件夹。")