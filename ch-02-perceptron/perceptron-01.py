
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 构造训练数据集
    x_train = np.array([[3,3], [4,3],[1,1]])
    y = np.array([1,1,-1])

    # 构建感知机对象，对数据进行训练
    perceptron = MyPerceptron(None,0)
    perceptron.train(x_train, y)

    # 绘制图像
    draw(x_train, perceptron.w, perceptron.b)

# 构建 MyPerceptron 感知机 类
class MyPerceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.l_rate = 1

    # 核心算法实现
    def train(self, x_train, y):

        """
            训练感知机模型
            :param x_train: 输入特征矩阵 (N, d)，每行是一个样本
            :param y: 标签向量 (N, )，每个样本的标签（+1 或 -1）
         """
        # 用样本点的特征数据更新初始 w，如 x1 = (3,3), 有两个特征值，则 self.w = [0,0]
        # 初始化权重和偏置
        n_features = x_train.shape[1]
        self.w = np.zeros(n_features)
        i = 0
        while i < x_train.shape[0]:
            xi = x_train[i]
            yi = y[i]

            # 如果 y * (wx + b) ≤ 0 说明是误判点，更新 w，b
            if yi * (np.dot(self.w, xi) + self.b) <= 0:
                self.w = self.w + self.l_rate * yi * xi
                self.b = self.b + self.l_rate * yi
                print(f"Updated weights: {self.w}, bias: {self.b}")  # 调试信息
                # 如果是误判点，从头进行检测
                i = 0
            else:
                i += 1


def draw(x_train, w, bias):
    # 绘制数据点
    for i in range(x_train.shape[0]):
        if i == 0:
            plt.scatter(x_train[i][0], x_train[i][1], c='red', label='Positive Class (1)')
        elif i == 2:
            plt.scatter(x_train[i][0], x_train[i][1], c='blue', label='Negative Class (-1)')
        else:
            plt.scatter(x_train[i][0], x_train[i][1], c='red')

    # 判断 w[0] 和 w[1] 是否均为 0
    if w[0] == 0 and w[1] == 0:
        print("Warning: Both w[0] and w[1] are zero. No decision boundary can be drawn.")
        return

    if w[1] == 0:
        # 如果 w[1] 为 0，分离超平面为垂直线
        if w[0] != 0:
            x_vert = -bias / w[0]  # 计算垂直线的 x 坐标
            plt.axvline(x=x_vert, color='green', label='Decision Boundary')
    else:
        # 计算分离超平面
        x1 = np.linspace(0, 5, 100)
        x2 = -(w[0] * x1 + bias) / w[1]
        plt.plot(x1, x2, color='green', label='Decision Boundary')

    # 图像美化
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # 显示图像
    plt.show()


if __name__ == '__main__':
    main()