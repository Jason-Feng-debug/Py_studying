import numpy as np
from scipy.optimize import minimize

# 定义目标函数
def func(x, sign=1.0):
    # scipy.minimize默认求最小，求max时只需要sign*(-1)，跟下面的args对应
    return sign * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    # return sign * (np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

# 定义目标函数的梯度
def func_deriv(x, sign=1):
    jac_x0 = sign * (2 * x[0])
    jac_x1 = sign * (2 * x[1])
    jac_x2 = sign * (2 * x[2])
    return np.array([jac_x0, jac_x1, jac_x2])

# 定义约束条件
# constraints are defined as a sequence of dictionaries, with keys type, fun and jac.
cons = (
    {'type': 'eq',
     'fun': lambda x: np.array([x[0] + 2 * x[1] - x[2] - 4]),
     'jac': lambda x: np.array([1, 2, -1])},

    {'type': 'eq',
     'fun': lambda x: np.array([x[0] - x[1] - x[2] + 2]),
     'jac': lambda x: np.array([1, -1, -1])}
    )

# 定义初始解x0
x0 = np.array([-1.0, 1.0, 1.0])

# 使用SLSQP算法求解
res = minimize(func, x0 , args=(1,), jac=func_deriv, method='SLSQP', options={'disp': True},constraints=cons)
# args是传递给目标函数和偏导的参数，此例中为1，求min问题。args=-1时是求解max问题
print(res.x)

print(res.message)