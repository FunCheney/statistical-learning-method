import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        """
        KNN 类构造函数
        :param k: 选择邻居的数量，默认值为3
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        训练 KNN 模型
        :param X_train: 训练数据的特征矩阵 (n_samples, n_features)
        :param y_train: 训练数据的标签 (n_samples, )
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        对测试数据进行预测
        :param X_test: 测试数据的特征矩阵 (n_samples, n_features)
        :return: 预测的标签
        """
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        """
        对单个样本进行预测
        :param x: 单个测试样本的特征
        :return: 预测的标签
        """
        # 计算与训练数据的距离（使用向量化计算）
        distances = np.linalg.norm(self.X_train - x, axis=1)

        # 获取最近的 k 个样本的索引
        k_indices = np.argsort(distances)[:self.k]

        # 获取这 k 个样本的标签
        k_nearest_labels = self.y_train[k_indices]

        # 进行多数投票，选择最常见的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# 测试 KNN 实现
if __name__ == "__main__":
    # 构造简单的训练数据
    X_train = np.array([
        [5, 4],
        [9, 6],
        [4, 7],
        [2, 3],
        [8, 1],
        [7, 2]
    ])
    y_train = np.array([1, 1, 1, -1, -1, -1])

    # 创建 KNN 模型，选择 k=3
    knn = KNN(k=2)

    # 训练模型
    knn.fit(X_train, y_train)

    # 测试数据
    X_test = np.array([[5, 3]])

    # 预测测试数据的标签
    predictions = knn.predict(X_test)

    print("k={},被分类为：{}".format(1, predictions))
