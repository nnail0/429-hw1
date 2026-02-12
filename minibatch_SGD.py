# p4: minibatch_SGD

# %% [markdown]
# # Logistic Regression
# Nathan Nail - CS 429 Fall 2026
# 
# **NOTE** this model also uses gradient descent. 



# %%
import numpy as np

class LogisticRegressionGD:
    """
    Logistic regression (a classifier model) using GD to converge to 
    a solution. 

    Parameters:
    ------
    eta: learning rate
    n_iter: number of epochs
    random_state: int seeding the RNG. 
    batch_size: int representing the size of each minibatch. 

    Attributes
    ----------
    w_: 1D array containing the weights after training. 
    b_: Scalar value of the bias unit. 
    losses: A list containing the MSE loss values for each epoch. 
    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state=1, batch_size = 32):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.batch_size = batch_size

# %% [markdown]
# ### Fit the model
# 
# NOTE: np.T transposes the matrix. 

"""
mini-batch workflow:
- use a list comprehension to make a nested list. A list containing n batches. 
- determine the batch size.
for **each** epoch:
- shuffle the data
- create batches of specified size 
- train on the batches, adjusting the model for each batch

a reshuffle and new rebatching is required for each epoch. 
"""
def fit_mini_batch_sgd(self, X, y):
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc = 0.00, scale = 0.01, size = X.shape[1])
    self.b_ = np.float_(0.)
 
    n_batches = np.ceil(X.shape[0] / self.batch_size)

    # list containing n_batches number of mini-batches. 
    batch_list = [[] for i in range(n_batches)]

    for i in range(self.n_iter):

        # create a shuffled list of indices. 
        idx_list = np.random.shuffle([range(X.shape[0])])

        # make the mini_batches by pulling from X.
        for i in range(self.batch_size):                                # repeat _batchsize_ number of times
            for list_num in range(batch_list):                          # for each list:
                batch_list[list_num][i] = X[idx_list[list_num + i]]     # place a value from X into a minibatch. 


        for batch in batch_list:
            net_input = self.net_input(batch)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors)
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1-output)))) / len(batch)
            self.losses.append(loss)

    # TODO add in per-epoch error tracking. 

    return self


# %%
def net_input(self, X):
    """
    Calculate the net input.
    """
    return np.dot(X, self.w_) + self.b


def activation(self, z):
    """
    Logistic sigmoid activation
    
    :param self: A reference to this class
    :param z: the output. 
    """
    # 1. makes 1 a float
    return 1. /  (1./ + np.exp(-np.clip(z, -250, 250)))

def predict(self, X):
    # predict in class 1 if the log. sig. returns over the 0.5 threshold. 
    return np.where(self.activation(self.net_input(X) >= 0.5, 1, 0))





