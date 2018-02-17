It is a privte repository to host my research on Active learning


## How to use the activelearn library
It tries to automate the active learning workflow. Specify the initial number of samples to train on, the pool and test datasets and the functions to train, test, clear, save and restore models (which can be written in any framework such as keras or tensorflow).
	

    a = ActiveLearner(pool_data, pool_labels, test_data, test_labels, clear_model, train_fn, test_fn, save_model, restore_model, init_num_samples=100)

To the run function of the ActiveLearner instance, pass as an argument either a function to compute an acquisition operation over all pool data points, or a list of acquisition function references. You can also run for multiple experiments and compute the average using the experiment function.

	var_acc = a.experiment(100, [random_acq, var_ratio_tf], pool_subset_count=1000, num_exp=3)

`a.plot()` can be used to plot the accuracy curve. There is an option to pass the title or the labels as well. The plot function can be used by passing x axis and y axis explicitly.
