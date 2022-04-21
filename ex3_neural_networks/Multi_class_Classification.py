# https://github.com/kaleko/CourseraML/blob/master/ex3/ex3.ipynb

import random  # To pick random images to display

import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.io  # Used to load the OCTAVE *.mat files
import scipy
from scipy.special import expit  # Vectorized sigmoid function

datafile = 'ex3data1.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y']

# 显示X,y原始维度
# print("'y'shape: %s. Unique elements in y: %s"%(mat['y'].shape,np.unique(mat['y'])))
# print("'X'shape: %s. X[0] shape: %s"%(X.shape,X[0].shape))

# 往X处insert一列1
X = np.insert(X, 0, 1, axis=1)  # axis=0按行插入，axis=1是按列插入
# unique函数：对于一维数组或者列表去重并按元素由大到小返回一个新的无元素重复的元组或者列表
print("'y'shape: %s. Unique elements in y: %s" % (mat['y'].shape, np.unique(mat['y'])))
print("'X'shape: %s. X[0] shape: %s" % (X.shape, X[0].shape))
'''
图像在X中表示为400维向量(其中有5,000个).400维“特征”是原始20 x 20图像中每个像素的灰度强度。
类标签在向量y中作为表示图像中数字的数字类。
#X is 5000 images. Each image is a row. Each image has 400 pixels unrolled (20x20)
#y is a classification for each image. 1-10, where "10" is the handwritten "0"
'''


# Visualizing the data
def getDatumImg(row):
    """
    Function that is handed a single np array with shape 1x400,
    crates an image object from it, and returns it
    """
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


def displayData(indices_to_display=None):  # 索引to显示
    """
    Function that picks 100 random rows from X, creates a 20x20 image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indices_to_display:
        indices_to_display = random.sample(range(X.shape[0]), nrows * ncols)

    big_picture = np.zeros((height * nrows, width * ncols))

    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = getDatumImg(X[idx])
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1

    pyplot.imshow(big_picture, cmap="gray")
    pyplot.show()


displayData()


# Vectorizing Logistic Regression

def h(mytheta, myX):  # Logistic hypothesis function
    return expit(np.dot(myX, mytheta))


def commputeCost(mytheta, myX, myy, mylambda=0.):
    m = myX.shape[0]  # 5000
    myh = h(mytheta, myX)  # shape:(5000,1)
    term1 = np.log(myh).dot(-myy.T)  # shape:(5000,5000),dot是矩阵乘，*是对应位乘
    term2 = np.log(1 - myh).dot(1 - myy.T)  # shape:(5000,5000)
    left_hand = (term1 - term2) / m  # shape:(5000,5000)
    right_hand = mytheta.T.dot(mytheta) * mylambda / (2 * m)  # shape: (1,1)
    return left_hand + right_hand  # shape:(5000,5000)


# One-vs-all Classification
'''
# An alternative to OCTAVE's 'fmincg' we'll use some scipy.optimize function, "fmin_cg"
# This is more efficient with large number of parameters.
# In the previous homework, I didn't have to compute the cost gradient because
# the scipy.optimize function did it for me with some kind of interpolation...
# However, fmin_cg needs the gradient handed do it, so I'll implement that here
'''


def costGradient(mytheta, myX, myy, mylambda=0):
    m = myX.shape[0]
