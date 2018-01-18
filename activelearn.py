import numpy as np

 
def active_pick(acq_fn, num_samples, pool_data, pool_labels, pool_subset_count = 2000):
    """Inputs: Unsupervised data, an acquisition function and number of samples to return
    Output: The datapoints from unsupervised data which has the highest value as per the acquisition function
    """
    #unsup_data = np.array(unsup_data)
    if len(pool_data) < num_samples:
        raise Exception('pool data is exhausted')
        
    if pool_subset_count > len(pool_data):
        pool_subset_count = len(pool_data)

    #values = [acq_fn(x) for x in pool_data]
    pool_subset_random_index = np.random.choice(range(len(pool_data)), pool_subset_count, replace=False)
    X_pool_subset = pool_data[pool_subset_random_index]
    y_pool_subset = pool_labels[pool_subset_random_index]

    print('Search over Pool of Unlabeled Data')

    values = acq_fn(X_pool_subset)
    pos = np.argpartition(values, -num_samples)[-num_samples:]
    datapoints = X_pool_subset[pos]
    labels = y_pool_subset[pos]
    
    pool_data = np.delete(pool_data, (pool_subset_random_index[pos]), axis=0)
    pool_labels = np.delete(pool_labels, (pool_subset_random_index[pos]), axis=0)
    print("Picked " + str(num_samples) + " datapoints\nSize of updated unsupervised pool = " + str(len(pool_labels)) + "\n")

    return datapoints, labels, pool_data, pool_labels



init_pick(pool_data, pool_labels, num):
    '''Pick num number of datapoints from the unsupervised data pool
    Remove them from the pool and return the data.
    Returns chosen datapoints and the updated pool_data'''
    if len(pool_data) < num:
        raise Exception('pool data is too small')
        
    indices = np.random.choice(range(len(pool_data)), num, replace=False)
    datapoints = pool_data[indices]
    labels = pool_labels[indices]
    pool_data = np.delete(pool_data, indices, axis=0)
    pool_labels = np.delete(pool_labels, indices, axis=0)
    print("Picked " + str(num) + " datapoints\nSize of updated unsupervised pool = " + str(len(pool_data)) + "\n")
    return datapoints, labels, pool_data, pool_labels
