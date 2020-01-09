import numpy as np
from sklearn.metrics import accuracy_score
from classifier_c45 import *
from classifier_cart import *


def k_folds_cross_validation(x, y, num_folds = 10, classifier = 'c4.5'):

    #选择不同的算法
    if classifier == 'c4.5':
        classifiers = DTree_c45()
        name = 'c4.5'
    if classifier == 'cart':
        classifiers = DTree_cart()
        name = 'cart'

    x_train_folds = []
    y_train_folds = []
    x_train_folds = np.array_split(x, num_folds)
    y_train_folds = np.array_split(y, num_folds)

    acc = 0

    for i in range(num_folds):    
        index=i
        x_train_i = np.vstack((x_train_folds[0:i] + x_train_folds[i+1:num_folds]))
        y_train_i = np.hstack((y_train_folds[0:i] + y_train_folds[i+1:num_folds]))

        x_test_i = np.array(x_train_folds[index])
        y_test_i = np.array(y_train_folds[index])

        classifier = classifiers
        classifier.build(x_train_i, y_train_i)


        y_pred_i = classifier.predict_labels(x_test_i, classifier)
        accuracy = accuracy_score(y_test_i, y_pred_i)
        
        acc += accuracy
        print("{} accuracy: {}".format(name, accuracy))
   
    print("{} average_accuracy: {}".format(name, acc/num_folds) )
    return acc/num_folds