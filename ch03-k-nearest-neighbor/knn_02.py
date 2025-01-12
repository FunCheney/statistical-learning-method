# sklearn 实现 KNN

"""
: n_neighbors 近邻数  默认 5
: weights 近邻权重  默认：uniform 权重一样； distance 越近权重越大
: algorithm 算法   默认：auto 自动选择； brute 暴力求解；kd_tree KD 树；ball_tree 球树
        当数据量比较小的时候，不管设定那种算法，最终都会使用暴力求解的方法
: leaf_size 叶子节点数量的阈值，默认 30
: p 2
: metric  mincowski  欧式距离
: n_jobs  None 并行搜索，None 1 个进程，-1 所有进程
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def main():
    # 训练数据
    X_train = np.array([
        [5,4],
        [9,6],
        [4,7],
        [2,3],
        [8,1],
        [7,2]
    ])
    y_train = np.array([1,1,1,-1,-1,-1])

    # 待遇测数据
    X_new = np.array([[5,3]])

    for k in range(1, 6, 2):
        # 构建实例
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        # 选择合适的算法
        clf.fit(X_train, y_train)
        # 预测
        y_pred = clf.predict(X_new)
        print("k={},被分类为：{}".format(k, y_pred))

if __name__ == "__main__":
    main()