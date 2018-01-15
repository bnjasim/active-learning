import numpy as np

def active_pick(unsup_data, acq_fn, num_samples):
    """Inputs: Unsupervised data, an acquisition function and number of samples to return
    Output: The datapoints from unsupervised data which has the highest value as per the acquisition function
    """
    unsup_data = np.array(unsup_data)
    values = [acq_fn(x) for x in unsup_data]
    pos = np.argpartition(values, -num_samples)[-num_samples:]
    return unsup_data[pos]
    
