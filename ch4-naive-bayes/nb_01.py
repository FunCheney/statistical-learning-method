import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, lambda_):
        # 贝叶斯系数，取值为 0 时，即为极大似然估计
        self.lambda_ = lambda_
        # y的（类型：数量）
        self.y_types_count = None
        # y的（类型：概率）
        self.y_types_proba = None
        # （xi 的编号，xi 的取值，y的类型）：概率
        self.x_types_proba = dict()

    def fit(self, x_train, y_train):
        # y 的所有取值类型
        self.y_types = np.unique(y_train)
        # 转化为 pandas DataFrame 数据格式
        X = pd.DataFrame(x_train)
        y = pd.DataFrame(y_train)
        # y 的（类型：数量）统计
        self.y_types_count = y[0].value_counts()
        # y 的（类型：概率） 计算
        self.y_types_proba = (self.y_types_count + self.lambda_) / (y.shape[0] + len(self.y_types) * self.lambda_)

        # (xi 的编号，xi 的取值，y的类型)： 概率计算
        for idx in X.columns:  # 遍历 xi
            for j in self.y_types: # 选取每一个 y 的类型
                # 选择所有 y==j 为真的数据点的第 idx 个特征的值
                p_x_y = X[(y==j).values][idx].value_counts()
                print(p_x_y)
                # 计算（xi 的编号， xi 的取值， y的类型）: 概率
                for i in p_x_y.index:
                    print(i)
                    self.x_types_proba[(idx, i, j)] = (
                            (p_x_y[i] + self.lambda_) / (self.y_types_count[j] + p_x_y.shape[0] * self.y_types_count[j])
                    )


    def predict(self, x_new):
        res = []
        for y in self.y_types: # 遍历 y 的可能取值
            p_y= self.y_types_proba[y] # 获取 y 的概率 P(Y=ck)
            pxy = 1
            for idx, x in enumerate(x_new):
                pxy *= self.x_types_proba[(idx, x, y)] # 计算P(X=(x1,x2...xn)|Y=ck)

            res.append(p_y * pxy)

        for i in range(len(self.y_types)):
            print("[{}] 对应的概率：{:.2%}".format(self.y_types[i], res[i]))

        # 返回最大后验概率对应的 Y 值
        return self.y_types[np.argmax(res)]

def main():
    x_train = np.array([
        [1, "S"],
        [1, "S"],
        [1, "S"],
        [1, "S"],
        [1, "S"],
        [2, "S"],
        [2, "M"],
        [2, "M"],
        [2, "L"],
        [2, "L"],
        [3, "L"],
        [3, "M"],
        [3, "M"],
        [3, "L"],
        [3, "L"],
    ])

    y_train = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

    clf = NaiveBayes(lambda_= 0.2)

    clf.fit(x_train, y_train)
    x_new = np.array([2, "S"])
    y_pre = clf.predict(x_new)
    print("{} 被分类为：{}".format(x_new, y_pre))
    
if __name__ == '__main__':
    main()