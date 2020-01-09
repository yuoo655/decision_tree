import numpy as np
from load_data import load_data
import math
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DTree_cart:
    def __init__(self):
        self.best_feature = None
        self.bset_value = None
        self.bset_value_index = None
        self.left = None
        self.right = None
        self.value = None
        self.min_gini = 0

    def is_leaf(self):
        return self.value == True

    
    def gini(self, y):

        if len(y) == 0:
            return 1

        c = Counter(y)
        label_0 = c[0]
        label_1 = c[1]
        rate_0 = float(label_0/len(y))
        rate_1 = float(label_1/len(y))

        gini_ = 1 - rate_0**2 - rate_1**2

        return gini_

    def feature_gini(self, x, y, feature):
        """
        find feature's min_gini

        return split_value  min_gini
        """

        x = x[:, feature]
        y = y[np.argsort(x)]
        x = x[np.argsort(x)]

        #得到各种可能的划分值
        mid_x = (x[0:-1] + x[1:]) / 2

        gini_i = np.zeros(len(mid_x))

        #计算各种可能的划分方式的基尼指数
        for i in range(len(mid_x)):
            left_index = x[:] <= mid_x[i]
            right_index = x[:] > mid_x[i]

            y_left = y[left_index]
            y_right = y[right_index]

            rate_left = len(y_left)/ len(y)
            rate_right = len(y_right) / len(y)

            gini_left = self.gini( y_left)
            gini_right = self.gini( y_right)

            gini_i[i] = rate_left*gini_left + rate_right*gini_right

        best_mid_index = np.argmin(gini_i)
        best_mid = mid_x[best_mid_index]
        min_gini = np.min(gini_i)


        return best_mid, min_gini

    def split_feature_value(self, x, y):
        """
        找到当前样本最佳的划分方式

        返回最佳划分特征, 最佳划分值, 最小基尼指数
        """

        min_gini = 1
        best_feature = 0
        bset_value = 0

        num_features = x.shape[1]

        #寻找每个特征的最小基尼指数
        for i in range(num_features):            

            value , gini = self.feature_gini(x, y, i)

            if gini <= min_gini:
                min_gini = gini
                best_feature = i
                bset_value = value

        return best_feature, bset_value , min_gini
   
    def build(self, x, y):

        c = Counter(y)

        majority = c.most_common()[0] 

        most_common_label, count = majority[0], majority[1]

        # 创建叶节点
        if len(y) == count:
            self.value = most_common_label
            return

        if len(y) < 2 or len(c) == 1:
            self.value = most_common_label  
            return

        feature, value, min_gini = self.split_feature_value(x, y)

        # 创建叶节点
        if min_gini < 0.01:
            self.value = most_common_label
            return


        # split set
        left_data = x[:, feature] <= value
        right_data = x[:, feature] > value
        x1, y1, x2, y2 = x[left_data], y[left_data], x[right_data], y[right_data]


        # 划分后某一类数量为0 创建叶节点
        if len(y2) == 0 or len(y1) == 0:  
            self.value = most_common_label  
            return

        # 记录信息 递归build
        self.best_feature = feature
        self.bset_value = value
        self.min_gini = min_gini

        self.left = DTree_cart()
        self.right = DTree_cart()
        
        self.left.min_gini = min_gini
        self.right.min_gini = min_gini

        self.left.build(x1, y1)
        self.right.build(x2, y2)

    def sort(self, x):
        """
        为预测用
        """
        if self.value != None:
            return self.value
        if x[self.best_feature] <= self.bset_value:
            return self.left.sort(x)
        else:
            return self.right.sort(x)

    def predict_labels(self, x_test, tree):

        y_pred = [tree.sort(x_test[i]) for i in range(x_test.shape[0])]

        return y_pred



def run():
    x, y = load_data('data.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    tree = DTree_cart()
    tree.build(x_train, y_train)
    y_pred = tree.predict_labels(x_test, tree)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)

if __name__ == "__main__":
    run()
