#!/usr/bin/env python
import matplotlib.pyplot as plt
import pickle
import garf_stats as gs
import garf_plot as gp
import training_set as tset

# FIXME this should be converted into scaled and unscaled versions!!! ARGH!!!
master_test_set = tset.load_training_set('trsets/phone_box_section_2.tset')
unscaled_test_set = tset.MultiSequenceTrainingSet([master_test_set], [1])
scaled_test_set = tset.MultiSequenceTrainingSet([master_test_set])

results_dir = "results/"
run_title = "train_van_test_phone_13_"

error_plots = []
diags = []
forest_types = ["pairwise_singular"]
train_set_types = ["unscaled"]
for forest_type in forest_types:
    for training_set_type in train_set_types:
        for test_set, test_set_type in [(scaled_test_set, "scaled"), \
                                        (unscaled_test_set, "unscaled")]:
            fname = results_dir + run_title + forest_type + "_" + training_set_type + "_" + test_set_type + "_test.results"
            with open(fname) as f:
                results = pickle.load(f)

            print "loaded", forest_type, training_set_type, test_set_type
            gs.analyse_results(results)
            print ""
            error_plot = gp.OneDErrorPlot(results, test_set)
            error_plot.separate_uncertainty_axes = False
#            error_plot.draw_uncertainty = False
            error_plot.plot()
            error_plot.title("train on van, test on phone 13 with forest type %s, %s training set and %s test set." % \
                             (forest_type, training_set_type, test_set_type))
            [xmin, xmax, ymin, ymax] = plt.axis()
            plt.axis([xmin, xmax, -150, 150])
            error_plots.append(error_plot)

            diagnostics = gp.TsetDiagnostics()
            error_plot.attach_tset_idx_callback(diagnostics.show_data_point)
            diags.append(diagnostics)
