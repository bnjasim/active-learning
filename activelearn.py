import numpy as np


def var_ratio(pool_data):
    # Var ratio active learning acquisition function
    D_probs = model.predict_proba(pool_data)  
    return 1.0 - np.max(D_probs, axis=1)


def random_acq(pool_data):
    return np.random.rand(len(pool_data)) 


 
class ActiveLearner(object):
    '''Performs active learning
    acquisition_fn should return the prob to be acquired corresponding to each datapoint
    in the (subset of) pool data given as the argument'''
    
    
    def __init__(self, pool_data, pool_labels, test_data, test_labels, 
                 clear_model_fn, train_fn, eval_fn, 
                 save_model, recover_model,
                 init_num_samples=100):
        '''init_num_samples denote how many datapoints to be samples initially, 
        num_smaples denote how many datapoints to be samples at each iteration'''
        
        self.pool_data = pool_data
        self.pool_labels = pool_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.init_num_samples = init_num_samples
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        
        
        if (self.init_num_samples > len(pool_data)):
            raise Exception('Can not pick more samples than what is available in the pool data')
        
        # initialize empty arrays of dimension (0, 28, 28, 1) etc.
        self.train_data = np.empty((0,) + self.pool_data.shape[1:])
        self.train_labels = np.empty((0,) + self.pool_labels.shape[1:])
        
        self._accuracy = []
        self._x_axis = []
        self.acquisition_fn = object # just a placeholder for plot() function in case plot is called from outside
        
        # initial training. 
        # But make sure that the model is cleared of previous training
        clear_model_fn()
        self._init_pick()
        print ('Initial Training')
        self.train_fn(self.train_data, self.train_labels)
        # evaluate the accuracy after initial training
        self._accuracy.append(self.eval_fn(self.test_data, self.test_labels))
        self._x_axis.append(len(self.train_data))
        
        # save model function can be writing the learned model to the disk
        # or even taking a deep copy (in RAM)
        save_model()
        # we will recover the saved model in case of multiple runs
        self.recover_model = recover_model
    
    def _init_pick(self):
        '''Pick init_number_samples of datapoints from the unsupervised data pool
        for initial training of the model.
        Remove them from the pool and return the data.
        Returns chosen datapoints and the updated pool_data'''
        
        # This has already been checked in __init__
        if (self.init_num_samples > len(self.pool_data)):
            raise Exception('Can not pick more samples than what is available in the pool data')

        #np.random.seed(0)
        indices = np.random.choice(range(len(self.pool_data)), self.init_num_samples, replace=False)
        datapoints = self.pool_data[indices]
        labels = self.pool_labels[indices]
        self.pool_data = np.delete(self.pool_data, indices, axis=0)
        self.pool_labels = np.delete(self.pool_labels, indices, axis=0)
        print("Picked " + str(self.init_num_samples) + " datapoints\nSize of updated unsupervised pool = " +
              str(len(self.pool_data)) + "\n")
        
        if (len(self.train_data) > 0):
            raise Exception('In _init_pick: The train data is not empty.')
            
        self.train_data = np.vstack((self.train_data, datapoints))
        self.train_labels = np.vstack((self.train_labels, labels))
    
    
    def _active_pick(self):
        """Returns the datapoints which has the highest value as per the acquisition function
        from the pool_data
        """
        # This condition should ideally be False because we have already done 
        # the necessary checks while initializing run() function
        if (len(self.pool_data) < self.num_samples):
            raise Exception('Fatal mistake: pool data is exhausted')

        how_many = self.pool_subset_count if self.pool_subset_count <= len(self.pool_data) else len(self.pool_data)
        #values = [acq_fn(x) for x in pool_data]
        pool_subset_random_index = np.random.choice(range(len(self.pool_data)), how_many, replace=False)
        X_pool_subset = self.pool_data[pool_subset_random_index]
        y_pool_subset = self.pool_labels[pool_subset_random_index]

        print('Search over Pool of Unlabeled Data size = '+ str(len(X_pool_subset)))

        values = self.acquisition_fn(X_pool_subset)
        # pick num_samples of higest values in sorted (descending) order
        pos = np.argpartition(values, -self.num_samples)[-self.num_samples:]
        datapoints = X_pool_subset[pos]
        labels = y_pool_subset[pos]
        #print pool_subset_random_index[:10]
        self.pool_data = np.delete(self.pool_data, (pool_subset_random_index[pos]), axis=0)
        self.pool_labels = np.delete(self.pool_labels, (pool_subset_random_index[pos]), axis=0)
        print("\nPicked " + str(self.num_samples) + " datapoints\nSize of updated Unsupervised pool = " + 
              str(len(self.pool_data)))

        self.train_data = np.vstack((self.train_data, datapoints))
        self.train_labels = np.vstack((self.train_labels, labels))
    
    
    def run(self, n_iter, acquisition_fn, num_samples=10, pool_subset_count = None):
        '''Run active learning for given number of iterations'''
        
        self.num_samples = num_samples
        
        if ((pool_subset_count is None) or (pool_subset_count > len(self.pool_data))):
            self.pool_subset_count = pool_subset_count = len(self.pool_data)
        else:
            self.pool_subset_count = pool_subset_count 
        
        if (pool_subset_count < num_samples):
            raise Exception("pool subset count can't be smaller than num_samples")
            
        if (n_iter * num_samples > len(self.pool_data)):
            raise Exception('Pool data is small.\nReduce the number of iterations or number of samples to pick')
            
        self.acquisition_fn = acquisition_fn
        # self.n_iter = n_iter
        
        for i in range(n_iter):
            print('\nACQUISITION ITERATION ' + str(i+1) + ' of ' + str(n_iter))
            self._active_pick()
            self.train_fn(self.train_data, self.train_labels)
            self._accuracy.append(self.eval_fn(self.test_data, self.test_labels))
            self._x_axis.append(len(self.train_data))
        
        return self._x_axis, self._accuracy
            
    
    def plot(self, x_axis=None, y_axis=None, label=None, title='Active Learning', loc=0):
        '''Plot the accuracy'''
        if x_axis is None:
            x_axis = self._x_axis
            y_axis = self._accuracy
        
        %matplotlib inline
        from matplotlib import pyplot as plt
        
        x_start = self.init_num_samples
        x_end = x_axis[-1]
        y_start = np.round(np.min(y_axis), 1)
        y_end = 1.0
        
        if label is None:
            label = self.acquisition_fn.__name__

        plt.axis([x_start, x_end, y_start, y_end])
        # plt.yticks(np.array(range(11))*0.02 + 0.8)
        # plt.xticks(np.array(range(10))*100)
        plt.plot(x_axis, y_axis, label=label)
        plt.grid()
        plt.title(title)
        plt.legend(loc=loc)
        plt.show()