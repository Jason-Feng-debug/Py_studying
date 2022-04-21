import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as opt
import ex2_function as func

'''
scipy官方文档需要学习，Scipy 是一个用于数学、科学、工程领域的常用软件包，可以处理最优化、
线性代数、积分、插值、拟合、特殊函数、快速傅里叶变换、信号处理、图像处理、常微分方程求解器等
'''

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())

# 创建两个分数的散点图，样本为正则录取，样本为负则未被录取
# 正负样本分类
positive = data[data['Admitted'].isin([1])]  # 返回一个boolean，判断是否完全匹配
negative = data[data['Admitted'].isin([0])]
print('\n        正负样本分类如下：')
print(positive.head())
print(negative.head())

# 可视化正负样本
# subplots返回的值的类型为元组，其中包含两个元素：第一为画布，第二是子图
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o',
           label='Admitted')  # s是size，c是color，marker是标志
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend(loc=1)
ax.set_xlabel('Exam 1 Score')
ax.set_xlabel('Exam 2 Score')
plt.show()

# 绘制sigmoid激活函数
nums = np.arange(-10, 10, step=1)  # step为步长
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, func.sigmoid(nums), 'r')
plt.show()

# 编写代价函数评估结果，见函数

# 完成X，y，theta初始化矩阵设置
data.insert(0, 'Ones', 1)  # data中于0列添加1列1
cols = data.shape[1]  # 得到data的列数
print(cols)
X = data.iloc[:, 0:cols - 1]  # 暂时先理解为iloc为[3,4)左闭右开
y = data.iloc[:, cols - 1:cols]
print(X.head())
print(y.head())

# 将X和y转换为数组进行运算，初始化theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
X = np.matrix(X)

# 检查矩阵的维度，确保式子可乘
print('检查各个矩阵的维度，确保式子可乘')
print(' theta为：', theta)
print(' X的维度为：', X.shape, '\n', 'y的维度为：', y.shape, '\n', 'theta维度为：', theta.shape)

# 计算代价函数
cost = func.cost(theta, X, y)
print('初始化后的代价函数值为：\n', cost)

# 计算函数中一个梯度步长
grad1 = func.gradient(theta, X, y)
print('使用梯度函数计算的第一个梯度值为：\n', grad1)

# 使用SciPy's truncated newton（TNC）实现寻找最优参数
'''
func是所求的函数值，x0是最小值的初始解，fprime是函数的梯度
args是要传递给函数的参数；函数结果返回值为：最优解，函数迭代情况，返回代码。具体参考：
https://scipy.github.io/devdocs/reference/generated/scipy.optimize.fmin_tnc.html#scipy.optimize.fmin_tnc
'''
result = opt.fmin_tnc(func=func.cost, x0=theta, fprime=func.gradient, args=(X, y))
print('使用截断牛顿法所求的最优梯度为：\n', result[0])  # 函数结果返回值为：最优解，函数迭代情况，返回代码
theta = np.array(result[0])
print('截断法求的theta为：\n{0}'.format(theta))


# 绘制决策边界
func.plot_Decision_boundary(positive,negative,theta,X)

# 查看截断牛顿法所求出的代价函数计算结果cost值
cost_fmin_tnc = func.cost(result[0], X, y)
print('fmin_tnc函数下求到的代价函数值为：\n', cost_fmin_tnc)

# 现在知道了方程theta参数，参数theta来为数据集X输出预测。
# 然后，我们可以使用这个函数来给我们的分类器的训练精度打分。
# 当h(θ) >= 0.5时，预测 y=1;当h(θ) < 0.5时，预测 y=0.具体预测函数见ex2——function
theta_min = np.matrix(result[0])  # 提取fmin_tnc函数求的theta最优值
predictions = func.predict(theta_min, X)  # 求出预测的结果值
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
# a和b分别是预测值和实际值，a和b全0或全1则是预测正确
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象,具体参考：
# https://www.runoob.com/python3/python3-func-zip.html

# 学习变量打印方法，具体见：https://www.bilibili.com/video/BV1mj411f725?p=14
print(f'预测的样本总数为:{len(correct)},预测正确的样本总数为:{sum(correct)}\n预测具体结果分布如下(1为预测正确，0为预测错误)：\n{correct}')
# map函数不太懂，以下是计算精确度
accuracy = (sum(map(int, correct)) % len(correct))  # map() 会根据提供的函数对指定序列做映射；%是取模，返回除法的余数
print('预测的样本准确率Accuracy = {0}%'.format(accuracy))
# 我们的逻辑回归分类器预测正确，如果一个学生被录取或没有录取，达到89%的精确度。
# 不坏！记住，这是训练集的准确性。我们没有保持住了设置或使用交叉验证得到的真实逼近，
# 所以这个数字有可能高于其真实值（这个话题将在以后说明）。

