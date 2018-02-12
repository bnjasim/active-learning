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
        print ('Successfully saved the initial model')
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
    
    
    def _active_pick(self, acquisition_fn):
        """Returns the datapoints which has the highest value as per the acquisition function
        from the pool_data
        """
        # This condition should ideally be False because we have already done 
        # the necessary checks while initializing run() function
        if (len(self.pool_data) < self.num_samples):
            raise Exception('Fatal mistake: pool data is exhausted')
        
            
        how_many = self.pool_subset_count if self.pool_subset_count <= len(self.pool_data) else len(self.pool_data)
        pool_subset_random_index = np.random.choice(range(len(self.pool_data)), how_many, replace=False)
        X_pool_subset = self.pool_data[pool_subset_random_index]
        y_pool_subset = self.pool_labels[pool_subset_random_index]

        print('Search over Pool of Unlabeled Data size = '+ str(len(X_pool_subset)))

        values = acquisition_fn(X_pool_subset) # NOTE! This shouldn't be self.acquisition_fn
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
    
    
    def run(self, n_iter, acquisition_fn, num_samples=10, pool_subset_count = None, go_on=False):
        '''Run active learning for given number of iterations.
        The acquisition_fn is the name of the function which computes acquisition values.
        aquisition_fn can be a list of function references as well - which is called a multi_run.
        If go_on is True, the run can restart from where we left of'''
        
        self.num_samples = num_samples # required in _active_pick
        
        if ((pool_subset_count is None) or (pool_subset_count > len(self.pool_data))):
            self.pool_subset_count = pool_subset_count = len(self.pool_data)
        else:
            self.pool_subset_count = pool_subset_count 
        
        if (pool_subset_count < self.num_samples):
            raise Exception("pool subset count can't be smaller than num_samples")
            
        if (n_iter * self.num_samples > len(self.pool_data)):
            raise Exception('Pool data is small.\nReduce the number of iterations or number of samples to pick')
        
        # If aquisition_fn is a list - then multi_run. 
        if (type(acquisition_fn) is list):    
            # if type of aq function is list, then continue/go_on is not allowed
            if (go_on):
                # list implies a multi run so don't allow run to continue 
                raise Exception('Sorry! Continue is not allowed for multiple aquistion functions!')
                
            self._multi_run(n_iter, acquisition_fn)   
              
        else:
            # Single acquisition function  
            self._run(n_iter, acquisition_fn, go_on)
        
        return self._x_axis, self._accuracy    
    
    
    # This shouldn't be accessed direclty from outside, but only from run
    def _run(self, n_iter, acquisition_fn, go_on):
        # only a single acquisition function
        if (go_on):
            # Need to check if multi_run was not called just before!
            if type(self.acquisition_fn) is list:
                # list implies a multi run so don't allow run to continue 
                raise Exception('Sorry! A multi run was called before, can not continue from there!')

            print ('Continue iterations from where we left of last time\n')

        else:
            # Need to restore model and data     
            self._recover_model_and_data()
            
            # We can predefine the _x_axis 
            self._x_axis = [self.init_num_samples] # range(self.init_num_samples, self.init_num_samples + num_samples*(n_iter+1), num_samples)
            # initialize _accuracy matrix (2d array)
            # Do the testing with initial data
            # We could have saved that value, but it is a good check if the model is properly recovered or not
            self._accuracy = [self.eval_fn(self.test_data, self.test_labels)]

        self.acquisition_fn = acquisition_fn
        # self.n_iter = n_iter

        for i in range(n_iter):
            print('\nACQUISITION ITERATION ' + str(i+1) + ' of ' + str(n_iter))
            self._active_pick(acquisition_fn)
            self.train_fn(self.train_data, self.train_labels)
            self._accuracy.append(self.eval_fn(self.test_data, self.test_labels))
            self._x_axis.append(len(self.train_data))
    
    

    # This shouldn't be accessed direclty from outside, but only from run
    def _multi_run(self, n_iter, acquisition_fn):
        self.acquisition_fn = acquisition_fn
        
        # We can predefine the _x_axis 
        self._x_axis = range(self.init_num_samples, self.init_num_samples + self.num_samples*(n_iter+1), self.num_samples)
        # initialize _accuracy matrix (2d array)
        self._accuracy = np.zeros((len(acquisition_fn), len(self._x_axis)))

        for i_aq in range(len(acquisition_fn)):
            # recover the model
            self._recover_model_and_data()

            # Do the testing with initial data
            # We could have saved that value, but it is a good check if the model is properly recovered or not
            self._accuracy[i_aq, 0] = self.eval_fn(self.test_data, self.test_labels)

            for i in range(n_iter):
                print('\nFor Aquisition function: ' + str(acquisition_fn[i_aq].__name__) + ': ')
                print('ACQUISITION ITERATION ' + str(i+1) + ' of ' + str(n_iter))
                self._active_pick(acquisition_fn[i_aq])
                self.train_fn(self.train_data, self.train_labels)
                self._accuracy[i_aq, i+1] = self.eval_fn(self.test_data, self.test_labels)
                assert self._x_axis[i+1] == len(self.train_data)
                
    
    def _recover_model_and_data(self):
        self.recover_model()
        print ('Recovered Saved Model')
        # set train data to initially picked data only
        # reset pool_data to the whole data except initial data
        self.pool_data = np.vstack((self.pool_data, self.train_data[self.init_num_samples:]))
        self.pool_labels = np.vstack((self.pool_labels, self.train_labels[self.init_num_samples:]))
        self.train_data = np.delete(self.train_data, range(self.init_num_samples, len(self.train_data)), axis=0)
        self.train_labels = np.delete(self.train_labels, range(self.init_num_samples, len(self.train_labels)), axis=0)

        
    
    def plot(self, x_axis=None, y_axis=None, label=None, title='Active Learning', loc=0):
        '''Plot the accuracy'''
        %matplotlib inline
        from matplotlib import pyplot as plt
            
        if x_axis is None:
            x_axis = self._x_axis
            y_axis = np.array(self._accuracy)
        
        if len(x_axis) <= 1:
            raise Exception('First run before plotting!')
        
        x_start = x_axis[0] # self.init_num_samples
        x_end = x_axis[-1]
        y_start = np.round(np.min(y_axis), 1)
        y_end = 1.0
        if (y_start > np.min(y_axis)):
            y_start -= 0.1
        
        plt.axis([x_start, x_end, y_start, y_end])
        
        # if y_axis (_accuracy) is a matrix, not a 1d array
        if (len(y_axis.shape) > 1):
            # no label is given or label doesn't correpond to multi run
            if label is None or len(label) != len(y_axis):
                label = [s.__name__ for s in self.acquisition_fn]  

            for i in range(len(y_axis)):
                 plt.plot(x_axis, y_axis[i], label=label[i])   

        
        else:
            if label is None:
                acquisition_fn = self.acquisition_fn[0] if type(self.acquisition_fn) is list else self.acquisition_fn
                label = acquisition_fn.__name__

            plt.plot(x_axis, y_axis, label=label)
        
        plt.grid()
        plt.title(title)
        plt.legend(loc=loc)
        plt.show()  
        
    
    def experiment(self, n_iter, acquisition_fn, num_samples=10, pool_subset_count = None, num_exp=3):
        '''Run the experiments for the given number of times (num_exp) and 
        return average accuracy values over all experiments.'''
        
        if type(acquisition_fn) is not list:
            raise Exception('experiment is to compare different acquisition functions, hence it should be a list')
        
        # define new variables to hold averages: useful incase we want to stop experiment forcefully in between
        self._avg_accuracy = np.zeros((len(acquisition_fn), n_iter+1))
        
        for i in range(num_exp):
            print ('\nExperiment number : ' + str(i+1) + '\n****************\n')
            self.run(n_iter, acquisition_fn, num_samples, pool_subset_count)
            self._avg_accuracy = (self._avg_accuracy * (i) + self._accuracy) / (i+1) # running average
        
        # finally assign back the avg accuracy to _accuracy variable for proper plotting
        self._accuracy = self._avg_accuracy
        
        return self._avg_accuracy
            