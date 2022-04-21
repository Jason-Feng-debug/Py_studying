# 正则化逻辑回归,这个理论助于减少过拟合，提高模型的泛化能力。
# 数据可视化
# 有一些芯片在两次测试中的测试结果。对于这两次测试，你想决定是否芯片要被接受或抛弃。
# 为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，从其中构建一个逻辑回归模型。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
# print(f'Test 2数据\n{data2.iloc[:,1].head()}')
print(data2.head())

# 正负样本可视化
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend(loc=1)
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
ax.set_title('Component Quality Inspection')
plt.show()

# 创建多项式特征，即决策边界函数
'''
# 以下是测试循环输出的结果
degree = 5
for i in range(0,degree):
    for j in range(0,i):
        if j < degree-2:
            print(f'x{i}*x{j}', end='+')
        else:
            print(f'x{i}*x{j}')
# 输出结果为：x1*x0+x2*x0+x2*x1+x3*x0+x3*x1+x3*x2+x4*x0+x4*x1+x4*x2+x4*x3
'''
degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)
t = ((1 + (degree - 1)) * (degree - 1)) / 2
print(f'\n该循环共有%d项x1^(i-j)*x2^j' % t)

for i in range(0, degree):
    for j in range(0, i):
        data2[f'x1^{i}*x2^{j}'] = np.power(x1, i - j) * np.power(x2, j)
# data2.drop('Test 1',axis=)#drop具体看帮助文档

print(data2.head())



