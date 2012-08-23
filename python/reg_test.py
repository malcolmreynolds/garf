import numpy as np
import pyublas #this is used behind the scenes, ignore warnings about this not being used!
import garf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

n = 100
test_size = 50
dims = 1

x = (np.random.rand(n, dims) * 2) - 1
x = np.sort(x, axis=0)
y = np.zeros((x.shape[0], 2 * x.shape[1]))
y[:, 0] = x.flatten()
y[:, 1] = -4 * (x.flatten() ** 2) + 1
y = y + 0.15 * np.random.randn(y.shape[0], y.shape[1])

xtest = (np.random.rand(test_size, dims) * 4) - 2
ytest = np.zeros((xtest.shape[0], 2 * xtest.shape[1]))
ytest[:, 0] = xtest.flatten()
ytest[:, 1] = -4 * xtest.flatten() + 1
y_pred = np.zeros_like(ytest)
y_var = np.zeros_like(y_pred)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y[:, 0], y[:, 1], c='b')
ax.set_xlabel('X')
ax.set_ylabel('Y0')
ax.set_zlabel('Y1')

print "before training, covariance"


frst = garf.RegressionForestHyp()
frst.params.bagging = True
frst.params.min_sample_count = 5
frst.params.max_num_trees = 10
frst.params.max_tree_depth = 8
frst.params.num_threshes_per_split = 100

# frst.train(x, y)

# frst.predict(xtest, y_pred, y_var)


def blah():
    frst.train(x, y)

    frst.predict(xtest, y_pred, y_var)

    ax.scatter(xtest, y_pred[:, 0], y_pred[:, 1], c='r')
    y_min = y_pred - np.sqrt(y_var)
    y_max = y_pred + np.sqrt(y_var)
    for i in xrange(xtest.shape[0]):
        ax.plot([xtest[i], xtest[i]], \
                [y_min[i, 0], y_max[i, 0]], \
                [y_pred[i, 1], y_pred[i, 1]], c='r')
        ax.plot([xtest[i], xtest[i]], \
                [y_pred[i, 0], y_pred[i, 0]], \
                [y_min[i, 1], y_max[i, 1]], c='r')

blah()
            
