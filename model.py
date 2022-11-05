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

def loopGradientDescent(alpha, y, x, cnt):
	'''
	loopGradientDescent()函数，用来循环梯度下降求最小的损失函数
	输入:
		 alpha学习率，可以考虑采用动态步长
		 y样本值(m*1的一个数组，表示m个样本值)
		 x样本(m*(n+1)的一个数组，表示m个样本，n+1个特征)
		 cnt最大迭代次数
	输出:模型theta((n+1)*1的一个数组，表示n+1个特征的值)
	'''
	n = np.shape(x)[1]  #获取特征数量n+1
	m = np.shape(x)[0]  #获取样本数量m
	theta = np.mat(np.ones((n,1)))  #初始化训练模型
	for i in range(cnt):
		#print(np.shape(theta),np.shape(x))
		fc = forecast(theta,x)    #获取预测值矩阵(m*1)
		#这里直接使用矩阵乘法一次性计算出来(y-fc)*x矩阵(n*1)
		theta = theta + alpha*((x.T)*(y-fc)).T    #化简后的梯度下降
		if i % 100 == 0:    #每100次看一下效果
			errrate = errorRate(y,fc)
			print("第"+str(i)+"次循环,误差率:"+str(errrate))
			print("theta:")
			print(theta)
	return theta


def loadData():
    '''
    loadData()从训练集中读出数据，并分类set0和set1
    输入:fileName文件名
    输出:两个m*n的矩阵，m个样本，n个特征
    '''
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
    for i in range(len(test_y)):
        print("real %d, pred %d " % (test_y[i], y_pred[i]))
    print('测试集打分', result)
    print('训练集打分', classfier.score(train_x,train_y))
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


    

def saveModel(fileName,theta):
        '''
        saveModel()将模型保存到fileName的文件中。
        输入:fileName保存的文件名吗，一个字符串
             theta模型(n*1)的一个矩阵
        '''
        pass

def loadModel(fileName):
	'''
        loadModel()函数，根据文件名提取出模型theta，并返回一个n*1的矩阵
	输入:fileName文件名
	输出:一个(m*n)的矩阵,作为模型
		 一般来说，线性回归的theta模型是n*1的矩阵。
		 我们训练后保存的是一个1*n的列表，因此需要读出来后转置一下
	'''
	pass


#-----------下面用一个简单的样例测试一下是否输出成功---------------
if __name__ == '__main__':
    loadData()
'''输出结果
[[0.73105858]
 [0.26894142]]
'''
