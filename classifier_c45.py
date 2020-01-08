import numpy as np
from load_data import load_data
import math
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DTree:
    def __init__(self):
        self.best_feature = None
        self.bset_value = None
        self.bset_value_index = None
        self.left = None
        self.right = None
        self.value = None
        self.max_info_gain = 0

    def is_leaf(self):
        return self.value == True

    
    def entropy(self, x, y):

        assert(len(y) != 0)

        label_1 = np.count_nonzero(y)
        label_0 = len(y) - label_1
        # c = Counter(y)
        # label_0 = c[0]
        # label_1 = c[1]
        rata_0 = float(label_0)/len(y)
        rata_1 = float(label_1)/len(y)
        entropy =0

        if label_0 == 0:
            entropy = - rata_1*np.math.log2(rata_1)
        elif label_1 == 0:
            entropy = - rata_0*np.math.log2(rata_0)
        else:
            entropy = -( rata_0*np.math.log2(rata_0) + rata_1*np.math.log2(rata_1))

        return entropy
    
    def feature_gain(self, x, y, feature):
        """
        find feature's max_gain

        return split_value  max_gain
        """

        x = x[:, feature]
        index = np.argsort(x)
        x = x[index]
        y = y[index]
        unique_x = np.unique(x)

        mid_x = [(unique_x[i] + unique_x[i+1])/2 for i in range(len(unique_x)-1)]
        mid_x = np.array(mid_x)

        assert(len(mid_x) != 0)
        entropy_i = np.zeros(len(mid_x))
        base_entropy = self.entropy(x, y)

        for i in range(len(mid_x)):
            # 划分为Dt- 和 Dt+
            value = mid_x[i]
            left_index  = x[:] <= value
            right_index = x[:] > value
            x1, y1, x2, y2 = x[left_index], y[left_index], x[right_index], y[right_index]

            ent_left = self.entropy(x1, y1)
            ent_right = self.entropy(x2, y2)
            rate_left = float( len(y1)/len(y))
            rate_right = float( len(y2)/len(y))

            # D(H) - D(H/A)
            entropy_i[i] = base_entropy - (rate_left*ent_left + rate_right*ent_right)


        best_mid_index = np.argmax(entropy_i)
        best_mid = mid_x[best_mid_index]
        max_gain = np.max(entropy_i)

        return best_mid, max_gain

    def split_feature_value(self, x, y):
        """
        找到当前样本最佳的划分方式

        返回最佳划分特征, 最佳划分值, 最大信息增益
        """

        max_infogain = 0
        best_feature = 0
        bset_value = 0

        num_features = x.shape[1]

        for i in range(num_features):            
            value , infogain= self.feature_gain(x, y, i)
            # print("当前计算属性: {} 此属性最大信息增益为: {:0.4f} 区间划分值为: {}".format(feature_names[i], infogain, XX[index][i]) )

            if infogain >= max_infogain:
                max_infogain = infogain
                best_feature = i
                bset_value = value

        return best_feature, bset_value , max_infogain
   
    def build(self, x, y):
        """
        造树
        """

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

        feature, value, max_info_gain = self.split_feature_value(x, y)

        # 创建叶节点
        if max_info_gain == 0:
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

        # splitting procedure
        self.best_feature = feature
        self.bset_value = value
        self.max_info_gain = max_info_gain

        self.left = DTree()
        self.right = DTree()
        
        self.left.max_info_gain = max_info_gain
        self.right.max_info_gain = max_info_gain

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
    tree = DTree()
    tree.build(x_train, y_train)
    y_pred = tree.predict_labels(x_test, tree)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy)

if __name__ == "__main__":
    run()

