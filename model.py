'''
考虑的方案：
1. 线性回归
2. 神经网络
'''

import numpy as np
import DataProc as dp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn import tree

def forecast(theta, x):
	'''
	forecast()函数，用来求x的预测值
	输入:theta预测模型((n+1)*1表示n+1个特征)
		x样本(m*(n+1)表示m行n+1列，m个样本，每个样本有n+1个特征，多出来的是作为常数项1)
	输出:forecastValue预测值(m*1)的一个数组
	'''
	forecastValue = 1.0/(1+np.exp(-(x*theta)))#直接向量点乘获取预测值矩阵
	return forecastValue

def errorRate(y, fc):
	'''
	errorRate()函数，用来求损失函数
	输入:y样本值(m*1的一个矩阵，表示m个样本的样本值)
        fc预测值(m*1的一个矩阵，表示m个样本的预测值)
	输出:损失函数值
	'''
	m = np.shape(y)[0]    #获得样本数量m
	sum_err = 0.0           #记录误差值之和
	for i in range(m):
		sum_err += (np.log(fc[i,0])*y[i,0])+(np.log(1-fc[i,0])*(1-y[i,0]))
	sum_err /= m
	return sum_err



def TrainModel():
    data = dp.SampleBalance()
    data = np.array(data)
    choose = int(len(data) * 0.8)
    print(data[0])
    train_x = data[:choose, 2:]
    train_x = train_x.astype('float64')
    train_y = data[:choose, 1]
    train_y = train_y.astype('int')
    test_x = data[choose:, 2:]
    test_x = test_x.astype('float64')
    test_y = data[choose:, 1]
    test_y = test_y.astype('int')

	# 决策树
    classfier = tree.DecisionTreeClassifier(criterion='entropy', random_state=53, 
				splitter='random',max_depth=4,min_samples_leaf=10,min_samples_split=10)
    classfier.fit(train_x, train_y)
    result = classfier.score(test_x, test_y)
    y_pred = classfier.predict(test_x)
    print('测试集打分', result)
    print('训练集打分', classfier.score(train_x,train_y))
    return classfier
''' 
    count = 0
    for i in range(len(test_y)):
        if test_y[i] == y_pred[i]:
            count += 1
        print("real %d, pred %d " % (test_y[i], y_pred[i]))
'''

'''
	# SVM
    classfier = NuSVC(kernel='rbf', gamma='scale', nu=0.01)
    classfier.fit(train_x, train_y)
    y_pred = classfier.predict(test_x)
    for i in range(len(test_y)):
        print("real %d, pred %d " % (test_y[i], y_pred[i]))
    print(classfier.fit_status_)
'''	
'''
	#逻辑回归，效果不好
	classfier = LogisticRegression(max_iter=10000)
    classfier.fit(train_x, train_y)
    y_pred = classfier.predict(test_x)
    for i in range(len(test_y)):
        print("real %d, pred %d " % (test_y[i], y_pred[i]))
'''


def TestResult(fileName):

	pass



if __name__ == '__main__':
    model = TrainModel()
    TestResult()
