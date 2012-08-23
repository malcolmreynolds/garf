import numpy as np
import _ply_utils as pu
from os.path import isfile
import pyublas  # Needed for _ply_utils
"""Write out the results of some prediction to a file we can examine
in Meshlab"""

# Use these
col_lookup = {'r': np.array([255, 0, 0], dtype=np.uint8), \
              'g': np.array([0, 255, 0], dtype=np.uint8), \
              'b': np.array([0, 0, 255], dtype=np.uint8), \
              'k': np.array([0, 0, 0], dtype=np.uint8), \
              'w': np.array([255, 255, 255], dtype=np.uint8), \
              'rg': np.array([255, 255, 0], dtype=np.uint8)}


def write_multiple_training_sets(filename, tsets, cols_to_use=['r', 'g', 'b', 'rg', 'w']):
    """Given a list of multiple training sets, write them all to a single
    ply file with a different colour for each."""

    label_range = np.r_[0:3]

    num_tsets = len(tsets)
    num_datapoints = sum([tset.num_samples() for tset in tsets])

    points = np.zeros((num_datapoints, 3))
    colours = np.zeros((num_datapoints, 3), dtype=np.uint8)

    idx = 0
    for tset, col in zip(tsets, cols_to_use[:num_tsets]):
        rng = np.r_[idx:idx + tset.num_samples()]
        print "rng=", rng
        points[rng, :] = tset.labels[:, label_range]
        colours[rng, :] = np.tile(col_lookup[col], (len(rng), 1))
        idx = idx + len(rng)

    write_points(filename, points, colours=colours)


def write_points(filename, points, colours=None, force_overwrite=False):
    """Writes 3 dimensional points (with optional colours) a ply file."""
    if isfile(filename) and not force_overwrite:
        raise ValueError("%s exists, pass force_overwrite=True to export anyway", filename)
    if points.shape[1] != 3:
        raise ValueError("points should be nx3")

    if colours is None:
        pu.write_ply(filename, points)
    else:
        if colours.shape == (3,):
            colours = np.tile(colours, (points.shape[0], 1))
        elif colours.shape != points.shape:
            raise ValueError("colours and points must be same shape")
        pu.write_ply_colour(filename, points, colours)


def write_linked_points(filename, points1, points2, \
                        cols1=None, cols2=None, \
                        line_col=None, \
                        force_overwrite=False):
    """Writes 2 sets of 3d points (with optional colours) as well as lines linking them. Line
    is grey by default"""
    if isfile(filename) and not force_overwrite:
        raise ValueError("%s exists, pass force_overwrite=True to export anyway", filename)
    if points1.shape[1] != 3:
        raise ValueError("3D data only.")
    if points1.shape != points2.shape:
        raise ValueError("Different points array shapes must match.")

    # Make the lines 50% grey by default
    if line_col == None:
        line_col = 127 * np.ones(3, dtype=np.uint8)

    if not cols1 and not cols2:
        # No colours provided, so do white
        pu.write_ply_linked_pairs(filename, points1, points2, line_col)
    else:
        # Colours can be provided either as a full nx3 matrix or a 3 element vector
        # If provided as a vector, expand to full matrix so we don't have to write
        # another variant of the same function in C++
        if cols1.shape == (3,):
            cols1 = np.tile(cols1, (points1.shape[0], 1))
        elif cols1.shape != points1.shape:  # full matrix provided, check it is right shape
            raise ValueError("cols1.shape must be either (3,) or same as points1")

        if cols2.shape == (3,):
            cols2 = np.tile(cols2, (points2.shape[0], 1))
        elif cols2.shape != points2.shape:  # full matrix provided, check it is right shape
            raise ValueError("cols2.shape must be either (3,) or same as points2")

        pu.write_ply_linked_pairs_colour(filename, points1, points2, cols1, cols2, line_col)
