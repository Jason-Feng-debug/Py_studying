# https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/code/ex3-neural%20network/ML-Exercise3.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

data = loadmat('ex3data1.mat')
print(data)
print('X的维度为：', data['X'].shape, '\ny的维度为：', data['y'].shape)


# 图像在martix X中表示为400维向量（其中有5,000个）。400维“特征”是原始20 x 20图像中每个像素的灰度强度。
# 类标签在向量y中作为表示图像中数字的数字类。


# 接下来将逻辑回归实现修改为完全向量化（即没有“for”循环）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义代价函数
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.reval().shape[1])  # theta shape(1,400)
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y  # 对象为matrix，*为矩阵乘法，X shape：（5000，400），theta.T shape(400,1),error shape:(5000,1)
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)  # X[:,i] shape:(5000,1)
        else:
            grad[i] = np.sum(term) / len(X) + (learningRate / len(X) * theta[:, i])

        return grad


# 向量化梯度函数
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y  # 对象为matrix，*为矩阵乘法，X shape：（5000，400），theta.T shape(400,1),error shape:(5000,1)

    # 前半部分为误差项，((X.T*error)/len(X)) shape:(400,1)，后半部分为正则化项，shape:(1,400)
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)  # error shape:(5000,1), X[:, 0] shape:(5000,1)
    # 详见：https://blog.csdn.net/qq_44444503/article/details/124167366
    return np.array(grad).ravel()


'''
以上已经定义了代价函数和梯度函数，接下来构建分类器,把分类器训练包含在
一个函数中，该函数计算10个分类器中的每个分类器的最终权重，
并将权重返回为k X（n + 1）数组，其中n是参数数量。
'''


# k X (n + 1) array for the parameters of each of the k classifiers

def one_vs_all(X, y, num_labels, learningRate):
    rows = X.shape[0]  # rows=5000
    params = X.shape[1]  # params=400

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))  # all_theta.shape:(10,401)

    # 在截距项的开头插入一列1
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)  # theta为(1,401)的array
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))  # 将y_i展开并重构为（rows,1）的数组

        # 最小化目标函数，使用scipy最优化函数minimize
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learningRate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x
    return all_theta


'''
这里需要注意的几点：
首先，我们为theta添加了一个额外的参数（与训练数据一列），以计算截距项（常数项）。
其次，我们将y从类标签转换为每个分类器的二进制值（要么是类i，要么不是）。
最后，我们使用SciPy的较新优化API来最小化每个分类器的代价函数。 如果指定的话，API将采用目标函数，
初始参数集，优化方法和jacobian雅可比矩阵（目标函数一阶偏导）函数。 然后将优化程序找到的参数分配给参数数组。
'''

# 接下来实现向量化的代码时正确写入所有矩阵,保证维度正确
rows = data['X'].shape[0]  # rows=5000
params = data['X'].shape[1]  # params=400
all_theta = np.zeros((10, params + 1))  # all_theta.shape:(10,401)
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)  # X.shape:(5000,401)
theta = np.zeros(params + 1)  # theta.shape:(401,)
y_0 = np.array([1 if label == 10 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))  # y_0.shape：(5000, 1)
print('y标签unique去重后如下：\n', np.unique(data['y']))

# 测试一下确保训练函数正确运行，并且得到合理的输出。
all_theta = one_vs_all(data['X'], data['y'], 10, 1)  # all_theta.shape:(10,401)


# 接下来使用训练完毕的分类器预测每个图像的标签。
def predict_all(X, all_theta):
    rows = X.shape[0]  # 5000
    params = X.shape[1]  # 400
    num_labels = all_theta.shape[0]  # 10
    X = np.insert(X, 0, values=np.ones(rows), axis=1)  # X.shape:(5000,401)按列插入

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)  # all_theta.shape:(401,10)

    # 对于每个训练样本,输出类标签为具有最高概率的类
    h = sigmoid(X * all_theta.T)  # h.shape:(5000,10)

    h_argmax = np.argmax(h, axis=1)  # 按行找出h数组的最大值索引
    h_argmax = h_argmax + 1  # 因为找出来的最大值索引从0开始，+1得到正确的标签预测
    return h_argmax, h_argmax


y_pred = predict_all(data['X'], all_theta)
correct = [1 if a.all() == b.all() else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
