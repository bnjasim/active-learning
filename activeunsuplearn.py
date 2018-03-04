import numpy as np

def random_acq(pool_data, step=None):
    return np.random.rand(len(pool_data)) 

class ActiveUnsupLearner(object):
    '''Performs active learning
    acquisition_fn should return the prob to be acquired corresponding to each datapoint
    in the (subset of) pool data given as the argument'''
    
    def __init__(self, pool_data, pool_labels, test_data, test_labels, 
                 clear_model_fn, train_fn, eval_fn, 
                 save_model, recover_model,
                 init_num_samples = 100
                ):
        '''init_num_samples denote how many datapoints to be samples initially, 
        num_smaples denote how many datapoints to be samples at each iteration'''
        
        self.pool_data = pool_data
        self.pool_labels = pool_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.init_num_samples = init_num_samples
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        
        self.experiment_no = 1 
        
        
        if (self.init_num_samples > len(pool_data)):
            raise Exception('Can not pick more samples than what is available in the pool data')
        
        # initialize empty arrays of dimension (0, 28, 28, 1) etc.
        self.train_data = np.empty((0,) + self.pool_data.shape[1:])
        self.train_labels = np.empty((0,) + self.pool_labels.shape[1:])
        
        self._accuracy = [] # keeps the accuracy after active+unsup(if required) picks
        self._accuracy_before = [] # accuracy just after active, before unsup pick, will be empty if no unsup
        self._x_axis = [] # keeps the number of labels queried + init_num_samples
        self._x_total = [] # keeps the total train_data size, will be empty if unsup mode is OFF
        
        self.acquisition_fn = None
        
        # initial training. 
        # Make sure that the model is cleared of previous training
        clear_model_fn()
        self._init_pick()
        print ('Initial Training')
        self.train_fn(self.train_data, self.train_labels, step=0)
        # evaluate the accuracy after initial training
        
        self._x_axis.append(len(self.train_data)) # same as init_num_samples
        self._accuracy.append(self.eval_fn(self.test_data, self.test_labels, step=len(self.train_data)))
        
        # save model function  writes the learned model to the disk
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

        # np.random.seed(0)
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
    
    
    def _active_pick(self, acquisition_fn, step=None):
        """Returns the datapoints which have the highest value as per the acquisition function
        from the pool_data
        """
        # This condition should ideally be False because we have already done the - 
        # necessary checks while initializing run() function
        if (len(self.pool_data) < self.num_samples):
            raise Exception('Fatal mistake: pool data is exhausted')
        
            
        how_many = self.pool_subset_count if self.pool_subset_count <= len(self.pool_data) else len(self.pool_data)
        pool_subset_random_index = np.random.choice(range(len(self.pool_data)), how_many, replace=False)
        X_pool_subset = self.pool_data[pool_subset_random_index]
        y_pool_subset = self.pool_labels[pool_subset_random_index]

        print('Search over Pool of Unlabeled Data size = '+ str(len(X_pool_subset)))

        # values = acquisition_fn(X_pool_subset, step) # NOTE! This shouldn't be self.acquisition_fn
        pos = acquisition_fn(X_pool_subset, self.num_samples, step) # NOTE! This shouldn't be self.acquisition_fn
        # pick num_samples of higest values in sorted (descending) order
        # pos = np.argpartition(values, -self.num_samples)[-self.num_samples:]
        
        # Instead of taking the 10 most uncertain values, why not take 100 most uncertain values 
        # and then pick 10 randomly? - Sorry that failed miserably for the MNIST!
        # num_to_pick = self.num_samples * 2
        # some heuristics - if we are picking more than half of pool_data, then may not be a good idea
        # if num_to_pick > how_many/2.0:
        #    num_to_pick /= 2
        
        # if still greater, then we probably want to use random sampling only
        # if num_to_pick > how_many:
        #    num_to_pick = how_many    
        # pos = np.random.choice(np.argpartition(values, num_to_pick)[num_to_pick:], self.num_samples, replace=False)
        
        datapoints = X_pool_subset[pos]
        labels = y_pool_subset[pos]

        self.pool_data = np.delete(self.pool_data, (pool_subset_random_index[pos]), axis=0)
        self.pool_labels = np.delete(self.pool_labels, (pool_subset_random_index[pos]), axis=0)
        print("\nActive Pick: Picked " + str(self.num_samples) + " datapoints")
        print("Size of updated Unsupervised pool = " + str(len(self.pool_data)))
        self.train_data = np.vstack((self.train_data, datapoints))
        self.train_labels = np.vstack((self.train_labels, labels))
        
        return len(pos)
    
    def _unsup_pick(self, step=None):
        '''Pick some datapoints from the unsupervised pool
        if the probability of one class is very high (like >0.99) 
        and add them to the train_data'''
        how_many = self.pool_subset_count if self.unsup_pool_size <= len(self.pool_data) else len(self.pool_data)
        pool_subset_random_index = np.random.choice(range(len(self.pool_data)), how_many, replace=False)
        X_pool_subset = self.pool_data[pool_subset_random_index]
        y_pool_subset = self.pool_labels[pool_subset_random_index]
    
        pos, labels = self.unsuper_fn(X_pool_subset, step)
        
        datapoints = X_pool_subset[pos]
        correct_labels = y_pool_subset[pos]
        
        if len(pos):
            unsup_mean = np.mean(np.argmax(labels, axis=1) == np.argmax(correct_labels, axis=1))
            print ('Unsupervise Picked: ' + str(np.sum(pos)) + ' samples')
            print ('Unsupervised: accuracy of labels = ' + str(unsup_mean))
        else:
            print ('Unsupervised: Nothing picked')

        self.pool_data = np.delete(self.pool_data, (pool_subset_random_index[pos]), axis=0)
        self.pool_labels = np.delete(self.pool_labels, (pool_subset_random_index[pos]), axis=0)
        # print("Size of updated Unsupervised pool = " +  str(len(self.pool_data)))

        self.train_data = np.vstack((self.train_data, datapoints))
        self.train_labels = np.vstack((self.train_labels, labels))
    
    def run(self, n_iter, acquisition_fn, unsuper_fn = None,
                 num_samples=10,  
                 pool_subset_count = None,
                 unsup_pool_size = None):
        '''Run active learning for given number of iterations.
        The acquisition_fn is(are) the name of the function(s) which computes acquisition values.
        aquisition_fn can be a list of function references as well - which is called a multi_run.
        '''
        
        self.num_samples = num_samples # required in _active_pick
        self.unsuper_fn = unsuper_fn
        self.unsup_pool_size = unsup_pool_size
        
        if ((pool_subset_count is None) or (pool_subset_count > len(self.pool_data))):
            self.pool_subset_count = pool_subset_count = len(self.pool_data)
        else:
            self.pool_subset_count = pool_subset_count 
        
        if (pool_subset_count < self.num_samples):
            raise Exception("pool subset count can't be smaller than num_samples")
            
        if (n_iter * self.num_samples > len(self.pool_data)):
            raise Exception('Pool data is small.\nReduce the number of iterations or number of samples to pick')
        
        # If aquisition_fn is a a single function reference
        if (type(acquisition_fn) is not list):    
            # if type of aq function is not list then make it a list
            acquisition_fn = [acquisition_fn]   
    
        self.acquisition_fn = acquisition_fn
        
        # self._x_axis = range(self.init_num_samples, self.init_num_samples + self.num_samples*(n_iter+1), self.num_samples)
        self._x_axis = np.zeros((n_iter + 1))
        self._x_total = np.zeros((len(acquisition_fn), len(self._x_axis)))
        
        # initialize _accuracy matrix (2d array)
        self._accuracy = np.zeros((len(acquisition_fn), len(self._x_axis)))
        self._accuracy_before = np.zeros((len(acquisition_fn), len(self._x_axis)))

        for i_aq in range(len(acquisition_fn)):
            # recover the model
            self._recover_model_and_data()
            
            # unsupervised pick only if unsup mode is ON
            if (self.unsuper_fn is not None):
                # keep the accuracy before unsup pick
                self._accuracy_before[i_aq][0] = self.eval_fn(self.test_data, self.test_labels, step=self.init_num_samples)

                self._unsup_pick(step=0)
                # Compute summaries over the pool_data
                # compute_pool_data_summary(self.pool_data)

                # Question - Do we have to retrain after unsup pick?
                # retrain only if we have picked at least 5 new datapoints by unsup pick
                # if ((len(self.train_data) - self.init_num_samples) >= 5):
                self.train_fn(self.train_data, self.train_labels)
                
                self._x_total[i_aq][0] = len(self.train_data)


            # accuracy is updated only after active+unsup pick
            self._accuracy[i_aq][0] = self.eval_fn(self.test_data, self.test_labels, step=len(self.train_data))
            self._x_axis[0] = self.init_num_samples
            

            for i in range(n_iter):
                print('\nExperiment ' + str(self.experiment_no) + ' Aquisition function: ' + str(acquisition_fn[i_aq].__name__) + ': ')
                print('ACQUISITION ITERATION ' + str(i+1) + ' of ' + str(n_iter))
                num = self._active_pick(acquisition_fn[i_aq], step=i)
                
                # if active_pick doesn't return even a single datapoint, stop
                if (num == 0):
                    break
                
                # allowing non fixed number of samples to be returned from active_pick
                # won't be suitable for running multiple experiments, but it will 
                # record only the last _x_axis
                self._x_axis[i+1] = self._x_axis[i] + num
                self.train_fn(self.train_data, self.train_labels)
                
                # unsupervised pick only if unsup mode is ON
                if (self.unsuper_fn is not None):
                    # keep the accuracy before unsup pick
                    self._accuracy_before[i_aq][i+1] = self.eval_fn(self.test_data, self.test_labels, step=len(self.train_data))

                    self._unsup_pick(step=0)
                    # Compute summaries over the pool_data
                    # compute_pool_data_summary(self.pool_data)

                    # Question - Do we have to retrain after unsup pick?
                    # retrain only if we have picked at least 5 new datapoints by unsup pick
                    # if ((len(self.train_data) - self.init_num_samples) >= 5):
                    self.train_fn(self.train_data, self.train_labels, step=i+1)

                    self._x_total[i_aq][i+1] = len(self.train_data)


                # accuracy is updated only after active+unsup pick
                self._accuracy[i_aq][i+1] = self.eval_fn(self.test_data, self.test_labels, step=len(self.train_data))
                
        return self._x_axis, self._accuracy 
    
    
    
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
            raise Exception('Please run experiment before plotting!')
    
        
        x_start = x_axis[0] # self.init_num_samples
        x_end = x_axis[-1]
        y_start = np.round(np.min(y_axis), 1)
        if (y_start > np.min(y_axis)):
            y_start -= 0.1
            
        # y_end = 1.0
        y_end = np.round(np.max(y_axis), 1)
        if (y_end < np.max(y_axis)):
            y_end += 0.1
        
        plt.axis([x_start, x_end, y_start, y_end])

        # no label is given or label doesn't correpond to multi run
        if label is None or len(label) != len(y_axis):
            if (self.acquisition_fn is None):
                raise Exception('Please pass the labels array as an argument')
            label = [s.__name__ for s in self.acquisition_fn]  

        for i in range(len(y_axis)):
             plt.plot(x_axis, y_axis[i], label=label[i])   

        
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
        self._avg_accuracy = np.zeros((len(acquisition_fn), n_iter+1)) # can't use zeros_like(self._accuracy) undef here
        self._avg_accuracy_before = np.zeros((len(acquisition_fn), n_iter+1))
        self._x_total_exp = np.empty((len(acquisition_fn), n_iter+1))
        
        for i in range(num_exp):
            self.experiment_no = i+1
            print ('\nExperiment #' + str(self.experiment_no) + '\n***************\n')
            self.run(n_iter, acquisition_fn, num_samples, pool_subset_count)
            self._avg_accuracy = (self._avg_accuracy * (i) + self._accuracy) / (i+1) # running average
            self._avg_accuracy = (self._avg_accuracy_before * (i) + self._accuracy_before) / (i+1) # running average
            self._x_total_exp = np.vstack((self._x_total_exp, self._x_total))
            
        # finally assign back the avg accuracy to _accuracy variable for proper plotting
        self._accuracy = self._avg_accuracy
        
        return self._x_axis, self._accuracy 
            