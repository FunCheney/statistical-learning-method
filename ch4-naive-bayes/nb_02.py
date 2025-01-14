# sklearn 实现 贝叶斯

import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn import preprocessing # 预处理

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

    y_train = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(x_train)
    x_train = enc.transform(x_train).toarray()
    print(x_train)
    clf = MultinomialNB(alpha=0.0000001)
    clf.fit(x_train, y_train)

    x_new = np.array([[2, "S"]])
    x_new = enc.transform(x_new).toarray()
    y_predict = clf.predict(x_new)
    print("{} 被分类为：{}".format(x_new, y_predict))
    print(clf.predict_proba(x_new))

if __name__ == "__main__":
    main()



