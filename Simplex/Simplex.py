import numpy as np


class Simplex:

    def __init__(self, A=np.empty([0, 0]), b=np.empty([0, 0]), c=np.empty([0, 0]), minmax="MAX"):  # 不指定默认为求MAX
        self.A = A
        self.b = b
        self.c = c
        self.x = [float(0)] * len(c)
        self.minmax = minmax
        self.printIter = True
        self.optimalValue = None
        self.transform = False  # 控制是否需要转换

    def addA(self, A):
        self.A = A

    def addB(self, b):
        self.b = b

    def addC(self, c):
        self.c = c
        self.transform = False

    def setPrintIter(self, printIter):
        self.printIter = printIter

    # 生成初始单纯性表
    def getTableau(self):
        # construct starting tableau
        # 当函数目标值为MIN时，将C值变为负值
        if self.minmax == "MIN" and self.transform == False:
            self.c[0:len(self.c)] = -1 * self.c[0:len(self.c)]
            self.transform = True

        # 构建初始单纯型表
        t1 = np.array([None, 0])  # t1
        numVar = len(self.c)  # 目标函数中原始变量数
        numSlack = len(self.A)  # 松弛变量数

        t1 = np.hstack(([None], [0], self.c, [0] * numSlack))  # hstack水平按列堆叠数组，构成初始单纯型表第一行Cj
        basis = np.array([0] * numSlack)  # 基变量，有几个松弛变量，添加几个0

        # 在原始变量的基础上增加基变量下标，便于输出基变量x……
        for i in range(0, len(basis)):
            basis[i] = numVar + i

        A = self.A

        if not ((numSlack + numVar) == len(self.A[0])):  # 创建松弛变量的基变量
            B = np.identity(numSlack)
            A = np.hstack((self.A, B))  # 将创建的基变量按列堆叠给A矩阵
        # # 测试A和b显示是否正常
        # print(A)
        # print(self.b)

        print('======================将原问题转化为标准形式如下：=============================')
        for i in range(0, len(A)):
            for j in range(0, len(A[0])):
                if j == len(A[0]) - 1:
                    print(f'{A[i, j]}*x{j + 1}', end=' = ')
                    print(self.b[i])
                else:
                    if A[i, j] >= 0:
                        print(f'{A[i, j]}*x{j + 1}', end=' + ')
                    else:
                        print(f'({A[i, j]})*x{j + 1}', end=' + ')  # x系数为“－”时，需要加上括号
        print("==========================================================================")

        t2 = np.hstack((np.transpose([basis]), np.transpose([self.b]), A))

        tableau = np.vstack((t1, t2))  # 把t1和t2按行堆叠

        tableau = np.array(tableau, dtype='float')

        # tableau[0,4] = 99

        return tableau

        # 输出标准函数式子

    def printOrigin(self):
        print('=========================原问题转化形式如下：================================')
        for i in range(0, len(self.A)):
            for j in range(0, len(self.A[0])):
                if j == len(self.A[0]) - 1:
                    print(f'{self.A[i, j]}*x{j + 1}', end=' <= ')
                    print(self.b[i])
                else:
                    if self.A[i, j] >= 0:
                        print(f'{self.A[i, j]}*x{j + 1}', end=' + ')
                    else:
                        print(f'({self.A[i, j]})*x{j + 1}', end=' + ')  # x系数为“－”时，需要加上括号
        print("==========================================================================")

    # 打印显示单纯型表
    def printTableau(self, tableau):

        print("变量\t", end="\t")
        print("最优值", end="\t\t")
        for j in range(0, len(self.c)):  # 打印初始变量
            print("x" + str(j + 1), end="\t\t\t")
        for j in range(0, (len(tableau[0]) - len(self.c) - 2)):  # 总列数-初始变量数-2（basis和c转置的列数）
            print("x" + str(j + len(self.c) + 1), end="\t\t\t")
        print()

        print('--------------------------------------------------------------------------')
        # 以下是输出tableau，按行打印
        for j in range(0, len(tableau)):  # len(tableau)为4行
            for i in range(0, len(tableau[0])):  # len(tableau[0])为7列
                if not np.isnan(tableau[j, i]):  # 判断是否是空值
                    if i == 0:
                        print(f"x{int(tableau[j, i]) + 1}", end="\t\t")  # 打印基变量所在列

                    else:

                        # # 尝试显示换入换出变量枢纽值
                        # if i == self.r and j == self.n:
                        #     print(f"[%.3f]" % tableau[j, i], end="\t\t")  # 此处添加小数位数，调整对齐
                        # else:

                        print(f"%.3f" % tableau[j, i], end="\t\t")  # 此处添加小数位数，调整对齐

                else:
                    print('σj  ', end="\t")  # 是空值时打印σj \t\t
            print()

    # 设置目标函数，求Max还是Min
    def setObj(self, minmax):
        if minmax == "MIN" or minmax == "MAX":
            self.minmax = minmax
        else:
            print("==========================================================================")
            print("          这是无效的函数目标，目标函数须是MAX或MIN，以下将默认求解MAX:")
            print("==========================================================================\n")
        self.transform = False

    # 执行优化迭代，直至生成最优解
    def optimize(self):
        # 判断目标函数，MIN与MAX相互转换（乘-1）
        if self.minmax == "MIN" and self.transform == False:
            for i in range(len(self.c)):
                self.c[i] = -1 * self.c[i]
                transform = True

        tableau = self.getTableau()
        if self.printIter:
            print("===========================初始化单纯型表如下:===============================")
            self.printTableau(tableau)

        # 当基变量不是最优时执行迭代优化
        optimal = False  # 初始基变量不是最优的，需要迭代换入换出
        iter = 1  # 便于查看每次迭代情况

        # 迭代运算，直到最优时停止迭代
        while True:

            if self.printIter:
                print("==========================================================================\n")
                print(f"============================第{iter}次迭代结果如下:===============================")
                self.printTableau(tableau)

            # 当目标函数是MAX时：检验数>0时需要旋转运算
            if self.minmax == "MAX":
                for profit in tableau[0, 2:]:  # tableau[0, 2:]为[2 3 0 0 0]
                    if profit > 0:
                        optimal = False
                        break  # 若profit>0，说明还未达到最优，break跳出该for循环，执行确定换入换出变量等后续操作
                    optimal = True

            # 当目标函数是MIN时：检验数<0时需要旋转运算
            else:
                for cost in tableau[0, 2:]:
                    if cost < 0:
                        optimal = False
                        break
                    optimal = True

            # 当所有检验数均满足条件时，即此时optimal = True，结束旋转运算
            if optimal:
                break

            # nth variable enters basis, account for tableau indexing
            # 第n个变量作为换入变量
            # index()返回指定值首次出现的位置。amax()返回数组的最大值或沿轴的最大值。
            if self.minmax == "MAX":
                # MAX：寻找检验数中的最大值作为换入变量
                n = tableau[0, 2:].tolist().index(np.amax(tableau[0, 2:])) + 2  # 2是上文的两列
            else:
                # MIN：寻找检验数中的最小值作为换入变量
                n = tableau[0, 2:].tolist().index(np.amin(tableau[0, 2:])) + 2

            # 最小比值原则确定换出变量，第r行变量作为换出变量
            minimum = 99999
            r = -1

            for i in range(1, len(tableau)):
                if tableau[i, n] > 0:  # tableau[i, n]指的是换入基那一列
                    val = tableau[i, 1] / tableau[i, n]
                    # 输出最小的比值的行，将这个值给r
                    if val < minimum:
                        minimum = val
                        r = i
            # 下面print是测试枢纽位置
            # print(r)
            # print(n)
            Simplex.r = r
            Simplex.n = n
            # 得到枢纽值为pivot
            pivot = tableau[r, n]
            print(f"换入变量位于该表格的第 {n + 1} 列")
            print(f"换出变量位于该表格的第 {r + 1} 行")
            print(f"交叉的枢纽值为[%.3f]" % pivot)

            # 转换矩阵，将基变量转换为单位阵
            # 换出变量所在行同时/pivot
            tableau[r, 1:] = tableau[r, 1:] / pivot

            # pivot other rows
            # 转换除换出变量以外的其他行
            for i in range(0, len(tableau)):
                if i != r:  # 换出变量所在行的其余行参与运算
                    mult = tableau[i, n] / tableau[r, n]
                    tableau[i, 1:] = tableau[i, 1:] - mult * tableau[r, 1:]

            # 更新基变量下标，生成新的基变量
            tableau[r, 0] = n - 2

            iter += 1

        # 输出最终计算的单纯型表
        if self.printIter:
            print("==========================================================================\n")
            print(f"========================最终单纯型表迭代{iter}次,结果如下：=========================")
            self.printTableau(tableau)
        else:
            print("问题已解决")

        # 保存问题的最优解

        # 输出最优解，生成所有变量的取值序列
        self.x = np.array([0] * (len(self.c)), dtype=float)
        # for i in range(0,len(self.c)+len(self.A)):
        #
        #
        #     print(self.x[i])

        for key in range(1, (len(tableau))):
            # 如初始变量x1,x2,下标最大2，所以只需要输出x1,x2即可,因此只需要保存初始函数变量的解就可以
            if tableau[key, 0] < len(self.c):
                self.x[int(tableau[key, 0])] = tableau[key, 1]
        #         因此只需按顺序赋值给对应的解值，就是该问题的解

        # 保存问题的最优值
        self.optimalValue = -1 * tableau[0, 1]  # 最优值*（-1）变换

    # 输出最优解和最优值
    def printSolu(self):

        print("==========================================================================")
        print(f"该问题最优解为:", end='')
        for i in range(len(self.x)):
            print(f"x%d" % (i + 1), end='')
            print(f"=%.3f" % self.x[i], end=' ')
        print(f"\n最优化目标值为:%.3f" % self.optimalValue)
