#!/usr/bin/env ipython --pylab -i
import numpy as np
import matplotlib.pyplot as plt
import cam3dutils.plot_cam as pc
import garf
import yep


def plot_error_bar_results(ax, x, y_mu, y_var, name, sqrt=True, alpha=0.5):

    if sqrt:
        y_var = np.sqrt(y_var)

    ax.scatter(x[:, 0], x[:, 1], y_mu, c=y_var, label=name)

    num_points = x.shape[0]

    error_bars = np.NaN * np.ones((3 * num_points, 3))
    error_bars[::3, 0] = error_bars[1::3, 0] = x[:, 0]
    error_bars[::3, 1] = error_bars[1::3, 1] = x[:, 1]

    error_bars[::3, 2] = y_mu[:, 0] + y_var[:, 0]
    error_bars[1::3, 2] = y_mu[:, 0] - y_var[:, 0]

    ax.plot(error_bars[:, 0], error_bars[:, 1], error_bars[:, 2],
            c='r', alpha=alpha, label='error_bars')


x = 18 * (np.random.rand(1000, 2) - 0.5).astype(np.float32)
thresh = 2
valid = ((x[:, 0] > thresh) |
         (x[:, 0] < -thresh) |
         (x[:, 1] > thresh) |
         (x[:, 1] < -thresh))
x = x[valid]

# # Generate simple regression problem, see if line improves
# x = np.vstack((2 * np.random.rand(100, 2),
#                np.array([2, 2]) + np.random.rand(100, 2),
#                np.array([2, -2]) + np.random.rand(100, 2),
#                np.array([-2, 2]) + np.random.rand(100, 2),
#                np.array([-1, -2]) + np.random.rand(100, 2))).astype(np.float32)
y = (x[:, 0] ** 2) - (x[:, 1] ** 2)
y = y.reshape((x.shape[0], 1))

noise = 3 * np.random.randn(x.shape[0]).reshape((y.size, 1))
y += noise


# Train two forests, one axis aligned and one hyperplanes
params = garf.make_params({'max_num_trees': 30,
                           'max_tree_depth': 6,
                           'num_splits_to_try': 20,
                           'balance_bias': 0.0})
yep.start('hyp.prof')
forest_hyp = garf.RegressionForestFltSmrtHyp(params)
forest_hyp.train(x, y)
yep.stop()


yep.start('ax.prof')
forest_ax = garf.RegressionForestFlt(params)
forest_ax.train(x, y)
yep.stop()

v = 10
test_x1, test_x2 = np.meshgrid(np.linspace(x[:, 0].min() - v,
                                           x[:, 0].max() + v, 25),
                               np.linspace(x[:, 1].min() - v,
                                           x[:, 1].max() + v, 25))
test_x = np.zeros((test_x1.size, 2), dtype=np.float32)
test_x[:, 0] = test_x1.flatten()
test_x[:, 1] = test_x2.flatten()

y_hyp_mu, y_hyp_var = forest_hyp.predict(test_x)
y_hyp_mu_now, y_hyp_var_now = forest_hyp.predict(test_x, use_weighted_average=False)
y_ax_mu, y_ax_var = forest_ax.predict(test_x)


ax = pc.get_3d_axis()
ax.scatter(x[:, 0], x[:, 1], y, c='r', label='raw data')
plot_error_bar_results(ax, test_x, y_hyp_mu, y_hyp_var, 'hyp')
plt.title('hyp')

ax2 = pc.get_3d_axis()
ax2.scatter(x[:, 0], x[:, 1], y, c='r', label='raw data')
plot_error_bar_results(ax2, test_x, y_ax_mu, y_ax_var, 'axis')
plt.title('axis')

ax = pc.get_3d_axis()
ax.scatter(x[:, 0], x[:, 1], y, c='r', label='raw data')
plot_error_bar_results(ax, test_x, y_hyp_mu_now, y_hyp_var_now, 'hyp')
plt.title('hyp no weight')

# Convert variance things into heatmap
fig = plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(np.log(y_hyp_var.reshape((25, 25))))
plt.colorbar()
plt.title('hyp')

plt.subplot(1, 3, 2)
plt.imshow(np.log(y_ax_var.reshape((25, 25))))
plt.colorbar()
plt.title('axis')

plt.subplot(1, 3, 3)
plt.imshow(np.log(y_hyp_var_now.reshape((25, 25))))
plt.colorbar()
plt.title('hyp no weight')


# hyp_root = forest_hyp.get_tree(0).root


