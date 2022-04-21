import numpy as np
import math

# 定义目标函数和偏导数
def Goal_func(x1, x2):
    # a = (3 / 2) * math.pow(x1, 2) + 0.5 * math.pow(x2, 2) - x1 * x2 - 2 * x1
    a = 1 * math.pow(x1, 2) + 2 * math.pow(x2, 2) - 2 * x1 * x2 - 4 * x1#PPT第二个例子
    return a


def Partial_x1(x1, x2):
    # b = 3 * x1 - x2 - 2
    b = 2 * x1 - 2 * x2 - 4
    return b


def Partial_x2(x1, x2):
    # c = x2 - x1
    c = 4 * x2 - 2 * x1
    return c

# G = [[3, -1], [-1, 1]]  # 目标函数的二阶偏导矩阵,海塞矩阵
G = [[2, -2], [-2, 4]]

# 初始化X值
X = [-9, 4]  # 初始点k=0

x = [X[0], X[1]]  # 保留初始点

Partial = [Partial_x1(X[0], X[1]), Partial_x2(X[0], X[1])]  # 一阶偏导
Search = [-Partial[0], -Partial[1]]  # 初始化搜索方向

Partial = np.matrix(Partial)
Search = np.matrix(Search)

# 计算初始步长,使用矩阵形式计算，计算最优步长alpha
alpha = -(Partial * Search.T) / (Search * G * Search.T)
print(f'初始化步长为：%.3f' % alpha)

Partial = [Partial_x1(X[0], X[1]), Partial_x2(X[0], X[1])]  # 一阶偏导向量
Search = [-Partial[0], -Partial[1]]  # 初始化搜索方向

# 已知初始步长和方向，可以求得下一点的X取值X1
X1 = [X[0] + alpha * Search[0], X[1] + alpha * Search[1]]

# 求出在X点处的梯度方向
Partial1 = [Partial_x1(X1[0], X1[1]), Partial_x2(X1[0], X1[1])]

# 计算Beta值的大小，从而确定从X1处出发的搜索方向
Beta = (math.pow(Partial1[0], 2) + math.pow(Partial1[1], 2)) / (math.pow(Partial[0], 2) + math.pow(Partial[1], 2))

# 设置函数计算的精确度，虽然共轭梯度求的最值是精确值，但是计算机计算有误差
# 设置Epsilon精确度，作为迭代条件
Epsilon = 0.0001

# 保存计算结果
compute_result = []
Minimum = Goal_func(X[0], X[1])

iter = 0

# 停止迭代验证方法，下降的梯度小于某个误差值,极值点的梯度下降为0
while (math.sqrt(math.pow(Partial[0], 2) + math.pow(Partial[1], 2)) > Epsilon):  # 最优点梯度模为0
    compute_result.append(Goal_func(X[0], X[1]))
    iter += 1

    if (Goal_func(X[0], X[1]) < Minimum):
        Minimum = Goal_func(X[0], X[1])

    # 更新函数取值X
    X[0] = X1[0]
    X[1] = X1[1]

    # 更新当前的梯度值
    Partial = [Partial_x1(X[0], X[1]), Partial_x2(X[0], X[1])]

    # 更新搜索方向P，为当前的负梯度方向
    Search = [-Partial[0], -Partial[1]]

    # 更新alpha搜索步长值
    alpha = -(Partial[0] * Search[0] + Partial[1] * Search[1]) / (
            (Search[0] * G[0][0] + Search[1] * G[1][0]) * Search[0] +
            (Search[0] * G[0][1] + Search[1] * G[1][1]) * Search[1])

    # 已知搜索方向和步长，确定下一点的X取值
    X1 = [X[0] + alpha * Search[0], X[1] + alpha * Search[1]]
    Partial1 = [Partial_x1(X1[0], X1[1]), Partial_x2(X1[0], X1[1])]
    # 更新beta值
    Beta = (math.pow(Partial1[0], 2) + math.pow(Partial1[1], 2)) / (math.pow(Partial[0], 2) + math.pow(Partial[1], 2))

# 输出结果
print("计算精度为{}情况下,迭代次数为：{}".format(Epsilon, iter))
print('==============================================')
print(f"设置的初始点为：(%.3f\n              %.3f)" % (x[0], x[1]))
print('==============================================')
print(f"求得的最优点为：(%.3f\n              %.3f)" % (X[0], X[1]))
print('==============================================')
min = Goal_func(X[0], X[1])
print(f"求得的最优值为：%.3f" % min)

