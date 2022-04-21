
import numpy as np

import Simplex
model = Simplex.Simplex()
#
# import M_Simplex
# model = M_Simplex.M_Simplex()

# 测试例子1
# A = np.array([[1, -2, 4, 2], [4, 0, -1, 1], [0, 4, 1, 2], [0, 5, -6, 0]])
# b = np.array([8, 16, 12, 9])
# c = np.array([-2, 3, -3, 6])


# 测试例子2
# A = np.array([[1, 2, 4], [4, 0, 1], [0, 4, 1], [0, 5, 6]])
# b = np.array([8, 16, 12, 9])
# c = np.array([-2, 3, 3])


# 测试例子3，已验证
# A = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1]])
# b = np.array([20, 20, 20])
# c = np.array([-10, -12, -12])



# 课本上例子
A = np.array([[1, 2], [4, 0], [0, 4]])
b = np.array([8, 16, 12])
c = np.array([-2, -3])

# # M_Simplex
# A = np.array([[1, -2, 1, 0], [-4, 1, 2, -1], [-2, 0, 1, 0]])
# b = np.array([11, 3, 1])
# c = np.array([3, -1, -1, 0])

model.addA(A)
model.addB(b)
model.addC(c)
model.setObj("MIN")  # 设置目标函数值是MIN或MAX
# print("A =\n", A, "\n")
# print("b =\n", b, "\n")
# print("c =\n", c, "\n\n")
model.printOrigin()
model.optimize()
model.printSolu()
print("==========================================================================")
