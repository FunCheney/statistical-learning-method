# 感知机 sklearn 实现

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
import numpy as np

# 构造训练数据集
x_train = np.array([[3,3], [4,3],[1,1]])
y = np.array([1,1,-1])

# 构建感知机对象，对数据进行训练 (正则化项，正则化系数，学习率，迭代次数，终止条件)
"""
    Q：L1, L2 分别有什么作用
        L1: 特征值更稀疏
        L2: 权值更均匀
    Q：正则化对系数的影响
        过大：无约束效力
        过小：约束的太狠
"""
perceptron = Perceptron(penalty="l1",alpha=0.1, eta0=0.1, max_iter=1000, tol=1e-3)
perceptron.fit(x_train, y)

print("w:", perceptron.coef_, "\n")
print("b:", perceptron.intercept_, "\n")
print("n_iter:", perceptron.n_iter_, "\n")

# 模型预测的准确率
res = perceptron.score(x_train, y)

print("correct rate:{:.0%}".format(res))