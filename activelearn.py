import numpy as np

def active_pick(unsup_data, acq_fn, num_samples):
    """Inputs: Unsupervised data, an acquisition function and number of samples to return
    Output: The datapoints from unsupervised data which has the highest value as per the acquisition function
    """
    unsup_data = np.array(unsup_data)
    values = [acq_fn(x) for x in unsup_data]
    pos = np.argpartition(values, -num_samples)[-num_samples:]
    return unsup_data[pos]


def init_pick(pool_data, pool_labels, num):
    '''Pick num number of datapoints from the unsupervised data pool
    Remove them from the pool and return the data.
    Returns chosen datapoints and the updated pool_data'''
    if len(pool_data) < num:
        raise Exception('pool data is too small')
        
    indices = np.random.choice(range(len(pool_data)), num, replace=False)
    datapoints = pool_data[indices]
    labels = pool_labels[indices]
    pool_data = np.delete(pool_data, indices)
    pool_labels = np.delete(pool_labels, indices)
    print("Picked " + str(num) + " datapoints\nSize of updated unsupervised pool = " + str(len(pool_labels)) + "\n")
    return datapoints, labels, pool_data, pool_labels
