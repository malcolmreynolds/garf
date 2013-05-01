from mpl_toolkits.mplot3d import Axes3D  # Needed! do not remove, despite
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import garf_stats as gs
import numpy as np
import optical_flow as of
import cam3dutils.homog_tform as ht


class TreeVisualiser(object):
    """Make a plot to visualise an individual tree"""
    # Thanks to Maciek for the original code for this stuff...
    # https://github.com/maciejgryka/draw_tree/
    def __init__(self, tree, fig=None, level_height=75, radius=10,
                 separation_at_bottom_level=0.5, max_line_width=50,
                 title=""):
        super(TreeVisualiser, self).__init__()
        self.fig = fig
        if fig is None:
            self.fig = plt.figure()
            self.fig.add_axes([0, 0, 1, 1])
        self.axes = self.fig.axes[0]

        self.level_height = level_height
        self.radius = radius
        self.separation_at_bottom_level = separation_at_bottom_level
        self.max_line_width = max_line_width

        # Do this in the constructor - doesn't make much sense to wait, and makes sure
        # there has been no change of figure focus since construction
        self.draw_tree(tree, title=title)

    def __del__(self):
        if self.fig is not None:
            plt.close(plt.figure.number)

    def draw_node(self, x, y, text, facecolor="#dddddd"):
        # print "circle at (%d,%d) radius %d" % (x, y, r)
        cir = mpatches.Circle((x, y), radius=self.radius, facecolor=facecolor)
        plt.text(x, y, text)
        self.axes.add_patch(cir)

    def plot_node_rec(self, node, x_offs, x_shift, parent=None):
        y_offs = -node.depth * self.level_height

        # If we are drawing a child node, draw a line to the parent with
        # thickness proportional to number of samples
        if parent is not None:
            thickness = max(self.max_line_width * float(node.samples.size) / self.amount_in_tree, 1.0)
            x_par_offs, y_par_offs = parent

            x_diff, y_diff = (x_par_offs - x_offs), (y_par_offs - y_offs)
            angle = np.arctan2(y_diff, x_diff)
            # Need to extend the lines to stop drawing on nodes
            x_remove = np.cos(angle) * self.radius
            y_remove = np.sin(angle) * self.radius

            self.axes.plot([x_offs + x_remove, x_par_offs - x_remove], \
                           [y_offs + y_remove, y_par_offs - y_remove], \
                           c='k', linewidth=thickness, alpha=0.5)

        self.draw_node(x_offs, y_offs, text=("%d:%d:%s" % (node.node_id,
                                                           node.samples.size,
                                                           str(node.dist.mu))))
        if node.is_leaf():
            # No children to recursively draw
            return

        self.plot_node_rec(node.l, x_offs - x_shift, x_shift / 2, parent=(x_offs, y_offs))
        self.plot_node_rec(node.r, x_offs + x_shift, x_shift / 2, parent=(x_offs, y_offs))

    def draw_tree(self, tree, title=""):
        """Entry function to all the plotting."""

        self.amount_in_tree = tree.root.samples.size
        x_shift_initial = self.separation_at_bottom_level * (2 ** tree.max_depth()) / 4
        self.plot_node_rec(tree.root, x_offs=0, x_shift=x_shift_initial)

        plt.axis('tight')
        plt.title(title)

    def savefig(self, filename):
        plt.figure(self.fig.number)
        plt.savefig(filename)


def switch_to_figure_or_create_new(fig):
    """If fig is a number then assume that indicates a figure and switch to it.
    Otherwise create a new one and return that."""
    if fig == None:
        return plt.figure()
    else:
        return plt.figure(fig)


def get_3d_axis(fig=None):
    """Get 3D axes on which to do plotting using the mplot3d library.
    If no figure is provided, create a new one."""
    fig = switch_to_figure_or_create_new(fig)
    return fig.add_subplot(111, projection='3d')


def plot_translational_errors(true_data, pred, pred_vars, \
                              subset=None, ax=None):
    """Given the true_data translations of various pairs in the dataset, plus
    the predicted outputs for each pair, plot some representation of the
    data. If the keyword argument subset is supplied it must be a numpy
    vector of indices which can be used to index into the data arguments
    and only plot a subset of them."""

    if ax == None:  # If no axis is provided, create a new one
        ax = get_3d_axis()

    if true_data.shape != pred.shape or \
       true_data.shape != pred_vars.shape:
        raise ValueError("shape mistmatch between true_data, predicted and pred_vars")

    num_data_points = true_data.shape[0]

    if subset == None:  # When no subset provided, plot all the data
        subset = np.r_[0:num_data_points]

    ax.scatter(true_data[subset, 0], true_data[subset, 1], true_data[subset, 2], c='b')
    ax.scatter(pred[subset, 0], pred[subset, 1], pred[subset, 2], c='r')

    # Join corresponding pairs of (true_data value, predicted value)
    link_corresponding_points(true_data, pred, ax, subset=subset)


class PlotInstance(object):
    """Just stores all the information about something user has clicked on"""
    def __init__(self, dataset, dataset_idx, seq, seq_idx_a, seq_idx_b, prediction, stddev, true_val):
        super(PlotInstance, self).__init__()
        self.dataset = dataset
        self.dataset_idx = dataset_idx
        self.seq = seq
        self.seq_idx_a = seq_idx_a
        self.seq_idx_b = seq_idx_b
        self.prediction = prediction
        self.stddev = stddev
        self.true_val = true_val


class OneDErrorPlot(object):
    """Allows us to plot some errors in a nice way."""
    def __init__(self, results, data_set):
        super(OneDErrorPlot, self).__init__()
        self.results = results
        self.data_set = data_set
        self.subset = None
        self.fig = None
        self.draw_uncertainty = True
        self.separate_uncertainty_axes = True
        self.draw_stddev = True
        self.alpha = 0.2
        self.std_alpha = 0.1
        self.sorted = dict()
        self.connection_id = None

    def __del__(self):
        """Closes the figure, performs any other cleanup"""
        print "deleting figure"
        if self.fig is not None:
            if self.connection_id is not None:
                self.fig.canvas.mpl_disconnect(self.connection_id)
            plt.close(self.fig)

    def activate(self):
        plt.figure(self.fig.number)

    def title(self, title):
        self.activate()
        plt.title(title)

    def plot(self, subset=None):
        """Draw the error stuff"""
        if self.fig is None:
            self.fig = plt.figure()
        else:
            # Switch to the current figure and clear it
            plt.figure(self.fig)
            plt.clf()

        # If we are drawing a separate subplot for standard deviation
        if self.separate_uncertainty_axes:
            plt.subplot(2, 1, 1)  # Setup a 2 row 1 column plot, start at the top

        # Sort the data according to ascending order of true labels
        if subset is not None:
            self.indices = np.argsort(self.results.true_y.flatten())
            self.sorted['true'] = self.results.true_y.flatten()[self.indices]
            self.sorted['pred'] = self.results.pred_y.flatten()[self.indices]
            self.sorted['vars'] = self.results.var_y.flatten()[self.indices]
        else:
            self.indices = np.argsort(self.results.true_y[subset].flatten())
            self.sorted['true'] = self.results.true_y[subset].flatten()[self.indices]
            self.sorted['pred'] = self.results.pred_y[subset].flatten()[self.indices]
            self.sorted['vars'] = self.results.var_y[subset].flatten()[self.indices]

        # Need x_range for plotting
        x_rng = np.r_[0:len(self.indices)]
        plt.plot(x_rng, self.sorted['true'], c='b')
        plt.scatter(x_rng, self.sorted['pred'], c='r', alpha=self.alpha)

        if self.draw_uncertainty:
            if self.draw_stddev:
                uncertainty = np.sqrt(self.sorted['vars'])
            else:
                uncertainty = self.sorted['vars']

            pred_plus_uncertainty = self.sorted['pred'] + uncertainty
            pred_minus_uncertainty = self.sorted['pred'] - uncertainty
            for x in x_rng:
                plt.plot([x, x], [pred_plus_uncertainty[x], pred_minus_uncertainty[x]], c='m', alpha=self.std_alpha)
            plt.legend(('true', 'predicted', 'uncertainty'))
        else:
            plt.legend(('true', 'predicted'))

        plt.ylabel('mm')

        if self.separate_uncertainty_axes:
            plt.subplot(2, 1, 2)
            self.plot_separate_uncertainty()

    def plot_separate_uncertainty(self):
        """Do a separate plot showing uncertainty behaviour,
        intended to be below main graph."""
        data_to_plot = self.sorted['vars']
        x_rng = np.r_[0:data_to_plot.size]  # This is necessary for scatter to work
        if self.draw_stddev:
            data_to_plot = np.sqrt(data_to_plot)
        plt.scatter(x_rng, data_to_plot)
        plt.plot(x_rng, data_to_plot)

    def remove_callback(self):
        """Get rid of any callback attached to this plot, if there is one."""
        if self.connection_id is not None:
            self.fig.canvas.mpl_disconnect(self.connection_id)
            self.connection_id = None

    def attach_tset_idx_callback(self, custom_callback, surrounding_points=10):
        """Attach the given callback to our plot. Callback will be disconnected when we delete.
        The provided callback will be called with a list of instances found at that button
        """
        # Get rid of any callback we may already have
        self.remove_callback()

        def internal_callback(event):
            """This gets called when the user clicks on the plot. We need to extract
            whatever data is close to the clicked point, then work out what real data that maps to."""
            # Get all the data within a few pixels either side of the click. Let's get surrounding 10-15
            try:
                xdata = event.xdata

                if xdata is None:
                    # User clicked outside axes
                    return
                x_image_idx = np.round(int(xdata))

                # Bounds check
                if not(0 <= x_image_idx < self.indices.size):
                    return

                # This is the test point the user clicked closest to, but we will go a but above and below that
                rng_half_width = np.round(surrounding_points / 2)
                start_range = max(0, x_image_idx - rng_half_width)
                end_range = min(self.indices.size, x_image_idx + rng_half_width)
                actual_tset_indices = self.indices[np.r_[start_range:end_range]]

                print "actual_tset_indices =", actual_tset_indices

                # Build up a list in here of what to pass on to the callback
                clicked_points = []
                for idx in actual_tset_indices:
                    (photo_idx_a, photo_idx_b), seq = self.data_set.lookup_sample(idx)
                    pi = PlotInstance(dataset=self.data_set, dataset_idx=idx, seq=seq,
                                      seq_idx_a=photo_idx_a, seq_idx_b=photo_idx_b,
                                      true_val=self.results.true_y[idx],
                                      prediction=self.results.pred_y[idx],
                                      stddev=self.results.var_y[idx])
                    clicked_points.append(pi)

                custom_callback(clicked_points)
            except Exception as e:
                print "caught unexpected exception:", e

        # Need to save a reference to the above function or it could get GC'd?
        self.callback = internal_callback
        self.connection_id = self.fig.canvas.mpl_connect("button_press_event", internal_callback)


class TsetDiagnostics(object):
    """Use in conjunction with OneDErrorPlot"""
    def __init__(self):
        super(TsetDiagnostics, self).__init__()
        self.figs = []

    def __del__(self):
        for f in self.figs:
            plt.close(f)

    def show_data_point(self, results):

        for inst in results:
            photo_1 = inst.seq.main_camera_img(inst.seq_idx_a)
            photo_2 = inst.seq.main_camera_img(inst.seq_idx_b)
            (flowx, flowy) = inst.seq.optical_flow_mtx(inst.seq_idx_a, inst.seq_idx_b)

            info_fig = plt.figure()
            self.figs.append(info_fig)

            plt.subplot(2, 2, 1)
            plt.imshow(photo_1)
            plt.title('%08d.png' % inst.seq.lookup.main[inst.seq_idx_a])

            plt.subplot(2, 2, 2)
            plt.imshow(photo_2)
            plt.title('%08d.png' % inst.seq.lookup.main[inst.seq_idx_b])

            plt.subplot(2, 2, 3)
            plt.imshow(of.visualise(flowx, flowy))
            plt.title('flow between %d and %d' % (inst.seq.lookup.main[inst.seq_idx_a],
                                                  inst.seq.lookup.main[inst.seq_idx_b]))

            print "correct answer for dataset index %d (%d,%d): %f, predicted %f +/- %f" % \
                (inst.dataset_idx, inst.seq.lookup.main[inst.seq_idx_a], inst.seq.lookup.main[inst.seq_idx_b], \
                 inst.true_val, inst.prediction, inst.stddev)


# To do some kind of click on graph -> info thing, we need
# a) way to get clicks on graph - too wide, so we maybe want to return
#    all values inside a 1 (more?) pixel range
# b) Way to get from an index in the training set display thing
#    back to a pair in the graph. That means we need to save
#    sorting above..

def plot_forest_depth_hist(forest, title=None, fig=None):
    """Plot the distribution of leaf depths in a forest."""

    switch_to_figure_or_create_new(fig)

    depths = gs.get_leaf_depths(forest)
    min_depth, max_depth = min(depths), max(depths)

    plt.hist(depths, bins=(max_depth - min_depth + 1))
    if title == None:
        title = 'histogram of depth at leaf nodes'
    plt.title(title)


def plot_forest_samples_at_leaf_hist(forest, title=None, fig=None):
    """Plot the distribution of number of samples landing at each node in a forest."""

    switch_to_figure_or_create_new(fig)

    samples = gs.get_num_in_each_leaf(forest)
    min_samples, max_samples = min(samples), max(samples)

    plt.hist(samples, bins=(max_samples - min_samples + 1))

    if title == None:
        title = 'histogram of number of samples at each leaf node'
    plt.title(title)


def link_corresponding_points(d1, d2, ax, subset=None, col=(0, 0, 0, 0.2)):
    """For 3d point sets d1 and d2, plot lines between the matching
    points. col=(0, 0, 0, 0.5), the default line colour, is black
    but half transparent."""
    if subset is None:
        subset = np.r_[:d1.shape[0]]
    for i in subset:
        ax.plot([d1[i, 0], d2[i, 0]], [d1[i, 1], d2[i, 1]], \
                [d1[i, 2], d2[i, 2]], c=col)


def plot_error_ranges(pred, pred_vars, ax, col=(1, 0, 0)):
    """Plot some representation of the uncertainty in the predicted results."""

    # Get data - variance and data + variance
    pred_min = pred - pred_vars
    pred_max = pred + pred_vars

    for i in xrange(pred.shape[0]):
        # Need 3 plots here to draw the lin
        ax.plot([pred_min[i, 0], pred_max[i, 0]], \
                [pred[i, 1], pred[i, 1]], \
                [pred[i, 2], pred[i, 2]], c=col)

        ax.plot([pred[i, 0], pred[i, 0]], \
                [pred_min[i, 1], pred_max[i, 1]], \
                [pred[i, 2], pred[i, 2]], c=col)

        ax.plot([pred[i, 0], pred[i, 0]], \
                [pred[i, 1], pred[i, 1]], \
                [pred_min[i, 2], pred_max[i, 2]], c=col)


def plot_error_histogram(true_data, pred, fig=None, bins=100):
    """Given some true 3d positions and predicted 3d positions, plots
    a histogram of squared error distribution."""
    # If we are not provided with a figure, make one,
    # otherwise activate the one we are provided with.
    if fig == None:
        fig = plt.figure()
    else:
        plt.figure(fig)
    # Calculate the errors
    squared_errors = gs.squared_error_vector(true_data, pred)

    # Returns
    (n, bins, patches) = plt.hist(squared_errors, bins=bins)
    return (n, bins, patches)


def draw_cam_frustum(ax, inverse_ext, label_string=None, aspect_ratio=4 / 3.0, scale=40, dotted=False):
    """Draw a traditional camera frustum / rectangular pyramid ting to show where
    the camera is."""
    # The NaN column is necessary because to plot a bunch of lines that aren't all
    # contiguous
    a = aspect_ratio
    points = np.array([[0, a,  a, -a, -a, np.NaN],
                       [0, 1, -1, -1,  1, np.NaN],
                       [0, 1,  1,  1,  1, np.NaN]])
    points *= scale

    # Indices to draw all the lines. Include an index of 5 to get a NaN
    # which allows us to start a new line.
    indices = [0, 1, 5, 0, 2, 5, 0, 3, 5, 0, 4, 5, 1, 2, 3, 4, 1]

    # import pdb; pdb.set_trace()
    pts_transformed = ht.apply(inverse_ext, points)
    pts_to_plot = np.asarray(pts_transformed[:, indices])
    col = "k-"  # solid line by default
    if dotted:
        col = 'k--'
    ax.plot(pts_to_plot[0, :], pts_to_plot[1, :], pts_to_plot[2, :], col)
    if label_string:
        ax.text(pts_to_plot[0, 0], pts_to_plot[0, 1], pts_to_plot[0, 2], label_string)


def draw_feature_visible_line(ax, homog_direction, inverse_ext, label_string=None, length=2000):
    """Given some 3D axes a direction for the ray, plus the extrinsics
    matrix, plot the ray in 3D"""
    coords = np.hstack((np.zeros((3, 1)), length * homog_direction.reshape(3, 1)))
    coords = ht.apply(inverse_ext, coords)
    if label_string:
        ax.plot(coords[0, :], coords[1, :], coords[2, :], 'b', label=label_string)
    else:
        ax.plot(coords[0, :], coords[1, :], coords[2, :], 'b')


def draw_axes_spikes(ax, ext, scale=50):
    s = scale
    pts = np.array([[0, s, 0, 0],
                    [0, 0, s, 0],
                    [0, 0, 0, s]])
    pts_tformed = ht.apply(ext, pts)

    for idx, col in zip([[0, 1], [0, 2], [0, 3]], ['r-', 'g-', 'b-']):
        ax.plot(pts_tformed[0, idx], pts_tformed[1, idx], pts_tformed[2, idx], col)

## Shit below here is OOOOLD, no guarantee it will work.


def plot_tree(t):
    nodes_to_visit = [t.root]
    # dimensionality = len(t.root.dist.mu.shape)
    print "plotting tree of dimensionality 1"

    while nodes_to_visit:
        n = nodes_to_visit.pop()
        try:
            l, r = n.l, n.r
            depth = n.depth
            #if we are here then the node is internal. Draw the tree shape
            x = np.array([l.dist.mu[0], n.dist.mu[0], r.dist.mu[0]])
            y = np.array([depth + 1, depth, depth + 1])
            print "x=", x
            print "y=", y
            plt.plot(x, y)
            nodes_to_visit.append(l)
            nodes_to_visit.append(r)
        except AttributeError:
            # do nothing - this just means we have hit a root node
            pass

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'w']


def plot_samples(t, depth, training_x, training_y):
    nodes_to_visit = [t.root]
    col_idx = 0
    print "col_idx =", col_idx, "colours[col_idx] =", colours[col_idx]

    while nodes_to_visit:
        n = nodes_to_visit.pop()
        if n.depth == depth:
            indices_here = n.samples
            print "plotting ", indices_here, "in", colours[col_idx]
            plt.scatter(training_x[indices_here, :], training_y[indices_here, :], c=colours[col_idx])
            col_idx = (col_idx + 1) % len(colours)
        else:
            nodes_to_visit.append(n.l)
            nodes_to_visit.append(n.r)
