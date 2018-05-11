
# coding: utf-8


# Code originally in Python 2.7 but works in python3 as well
# specifically train_data is a zip object in python3 as opposed to a list
# use list(train_data) to unpack 

import tensorflow as tf
import numpy as np
import os
import time


# In[2]:

# Adapted for active learning from: https://github.com/guillaumegenthial/sequence_tagging
# You need to copy the required files and dataset to the directory inorder to run this code

from model.config import Config
config = Config()
# config.use_crf = False
# config.lr_method = 'sgd'
config.nepochs = 10
config.lr_method = 'sgd'
config.lr_decay = 1.0


# In[3]:


config.filename_glove


# ### Load Data

# In[4]:


def load_data(filename):
    train_data, train_tags = [], []
    niter = 0
    with open(filename) as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    niter += 1
                    if config.use_chars:
                        words = zip(*words)
                    train_data += [words]
                    train_tags += [tags]
                    words, tags = [], []
            else:
                ls = line.split(' ')
                word, tag = ls[0],ls[-1]
                if config.processing_word is not None:
                    word = config.processing_word(word)
                    
                if config.processing_tag is not None:
                    tag = config.processing_tag(tag)
                words += [word]
                tags += [tag]
    
    return train_data, train_tags


# In[5]:


train_data, train_tags = load_data(config.filename_train)
test_data, test_tags = load_data(config.filename_test)
dev_data, dev_tags = load_data(config.filename_dev)

# in python3
# comment out in python2
train_data = [list(d) for d in train_data]
test_data = [list(d) for d in test_data]
dev_data = [list(d) for d in dev_data]


# In[6]:


# Random sampling of a batch of training data
def get_next_batch(data, labels=None, batch_size=None):
    '''data is a python list of shape (num_sentences) X (num_words_in_sent) X 2 (char_encoding + word_encoding tuple)
    We need to reshape it into (num_sentences) X 2 X (num_words_in_sent) '''
    # assert len(data) == len(labels), 'data and labels should be of the same size'
    
    if batch_size is None or batch_size > len(data):
        batch_size = len(data)
        
    indices = np.random.choice(range(len(data)), batch_size, replace=False)  
    batch_data = [data[i] for i in indices]

    if labels:    
        batch_labels = [labels[i] for i in indices]
        return batch_data, batch_labels
    else:  
        return batch_data


# ### Define Model and Train functions

# In[7]:


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger
        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings
        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None

        self.best_score = 0

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping
        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def restore_session(self):
        """Reload weights into session
        Args:
            sess: tf.Session()
            dir_model: dir with weights
        """
        print ("Reloading the latest trained model...")
        self.saver.restore(self.sess, self.config.dir_model)


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)


    def close_session(self):
        """Closes the session"""
        self.sess.close()


    def add_summary(self):
        """Defines variables for Tensorboard
        Args:
            dir_output: (string) where the results are written
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output, self.sess.graph)


    def train(self, train_data, train_tags, dev_data, dev_tags):
        """Performs training with early stopping and lr exponential decay
        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset
        """
        # best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        # self.add_summary() # tensorboard

        for epoch in range(self.config.nepochs):
            # self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

            score = self.run_epoch(train_data, train_tags, dev_data, dev_tags, epoch)
            # self.run_epoch(train_data, train_tags)
            # print ("score = " + str(score + " best score = "+ str(best_score)))
            self.config.lr *= self.config.lr_decay # decay learning rate

            # early stopping and saving best parameters
            if score > self.best_score:
                nepoch_no_imprv = 0
            #    self.save_session()
                self.best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv and epoch > 10: # minimum epochs
                    print ("- early stopping {} epochs without "                            "improvement".format(nepoch_no_imprv))
                    break


    def evaluate(self, test_data, test_tags):
        """Evaluate model on test set
        Args:
            test: instance of class Dataset
        """
        print ("Testing model over test set")
        metrics = self.run_evaluate(test_data, test_tags)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        print (msg)
        return metrics["f1"]


# In[8]:


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred
        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        # tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences
        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
            
            return labels_pred, sequence_lengths


    def run_epoch(self, train_data, train_tags, dev_data, dev_tags, epoch):
        """Performs one complete pass over the train set and evaluate on dev
        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch
        Returns:
            f1: (python float), score to select model on, higher is better
        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = 100 # (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i in range(nbatches):
            words, labels = get_next_batch(train_data, train_tags, batch_size)
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)
            _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            
            
        metrics = self.run_evaluate(dev_data, dev_tags)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        print (msg)

        return metrics["f1"]


    def run_evaluate(self, test_data, test_tags):
        """Evaluates performance on test set
        Args:
            test: dataset that yields tuple of (sentences, tags)
        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        
        batch_size = self.config.batch_size
        nbatches = (len(test_tags) + batch_size - 1) // batch_size
        
        # for words, labels in minibatches(test, self.config.batch_size):
        for i in range(nbatches):
            words = test_data[i*batch_size: (i+1)*batch_size]
            labels = test_tags[i*batch_size: (i+1)*batch_size]
            
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags
        Args:
            words_raw: list of words (string), just one sentence (no batch)
        Returns:
            preds: list of tags (string), one for each word in the sentence
        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds


# In[9]:


from model.data_utils import pad_sequences, get_chunks
from model.general_utils import Progbar


# In[10]:


config.nepoch_no_imprv = 5


# In[11]:


model = NERModel(config)
model.build()


# In[47]:


# model.train(train_data[1000:2000], train_tags[1000:2000], dev_data, dev_tags)


# In[48]:


# r = model.run_evaluate(test_data, test_tags)
# r


# ### Active Learning Experiments on NER

# In[13]:


# wrapper functions
def train_fn(data, labels, step=None):
    print ('Training data size: ' + str(len(data)))
    # start_time = time.time()
    config.nepochs = 10  
    model.train(data, labels, dev_data, dev_tags)
    # print ('Total time to train: ' + str(time.time() - start_time) + 's')
    

def test_fn(data, labels, step=None):
    print('Evaluate Model Test Accuracy after training')
    acc = model.evaluate(data, labels)
    return acc

clear_fn = model.initialize_session
save_fn = model.save_session
restore_fn = model.restore_session

def random_acq(pool_data, num_samples, step=None):
    # return np.random.rand(len(pool_data)) 
    return np.random.choice(len(pool_data), num_samples, replace=False)

out_prob_seq = tf.nn.log_softmax(model.logits)

def mnlp(pool_data, num_samples, step=None):
    # Var ratio active learning acquisition function
    # build feed dictionary
    feed, seq_len = model.get_feed_dict(pool_data, dropout=1.0)
    outprobs = model.sess.run(out_prob_seq, feed_dict=feed)
    
    outprobs = np.max(outprobs, axis=-1)
    
    for i in range(len(outprobs)):
        outprobs[i][seq_len[i]:] = 0
    
    probs = np.sum(outprobs, axis=1) / seq_len
    pos = np.argpartition(probs, num_samples)[:num_samples]
    return pos


# In[40]:


a = ActiveLearner(train_data, train_tags, test_data, test_tags, clear_fn, train_fn, test_fn, save_fn, restore_fn, init_num_samples=500)


# In[41]:


a.run(20, [mnlp, random_acq], pool_subset_count=1000, num_samples=20)


# In[42]:


a.plot()
