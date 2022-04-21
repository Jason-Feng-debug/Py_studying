import numpy as np
from matplotlib import pyplot as plt


# sigmoid激活函数。
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 逻辑回归的代价函数。
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    fitst = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(fitst - second) / (len(X))


# 执行批量梯度下降函数
# 实际上没有在这个函数中执行梯度下降，我们仅仅在计算一个梯度步长
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # 将theta reval，按列多维降为一维
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y  # 计算误差值

    for i in range(parameters):
        term = np.multiply(error, X[:, i])  # 计算代价函数的误差值
        grad[i] = np.sum(term) / len(X)  # 仅仅计算了第一步的梯度步长

    return grad


# 预测函数
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]  # 注意学习if—else语句缩写形式


# 当h(θ) >= 0.5时，预测 y=1;当h(θ) < 0.5时，预测 y=0.
# 输出的predictions如[0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1……]


# 绘制决策边界的函数
def plot_Decision_boundary(positive, negative, theta, X):
    # 任务：绘制决策边界,确定两个点，连成一条线
    '''
    Decision boundary occurs when h = 0, or when theta0 + theta1*x1 + theta2*x2 = 0
    #y=mx+b is replaced by x2 = (-1/theta2)(theta0 + theta1*x1)
    '''
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o',
               label='Admitted')  # s是size，c是color，marker是标志
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend(loc=1)
    ax.set_xlabel('Exam 1 Score')
    ax.set_xlabel('Exam 2 Score')

    boundary_xs = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 50)
    boundary_ys = (-1 / theta[2]) * (theta[0] + theta[1] * boundary_xs)
    ax.plot(boundary_xs, boundary_ys, 'r-', label='Decision Boundary')
    ax.legend(loc=1)
    plt.show()


# 正则化代价函数
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(1 - sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]]))
    return np.sum(first - second) / len(X) + reg


# 使用梯度下降法令正则化情况下代价函数最小化
def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # ravel将数组展开为一维数组
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)  # 不太理解为什么没有theta0-
        else:
            grad[i] = np.sum(term) / len(X) + ((learningRate / len(X)) * theta[:, i])  # 不太理解为什么没有theta-

    return grad
