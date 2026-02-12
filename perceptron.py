# %% [markdown]
# # Perceptron
# Nathan Nail, CS 429 Spring 2026
# 
# This code is adapted from the course's textbook. 
# 
# 

# %%
# perceptron.py
# Nathan Nail - CS 429 Spring 2026

# creates a perceptron classifier. 

import numpy as np

class Perceptron:
    """
    Docstring for Perceptron

    Hyperparameters:
    eta: a float representing the learning rate. 

    n_iter: Number of epochs to run through. 

    random_state: An integer used as the seed for the 
    random state generator. Initializes small values for
    best performance to prevent overcorrecting. 
    
    If all weights were 0, all will have the same error. 
    "Broken symmetry" allows for algorithm to work. 

    Attributes:
    Indicated with a _ afterwards as part of naming convention (how?
    to what end?)

    w_: vector of weights as a 1D array.

    b_: Scalar int, bias unit. 

    errors_: a list 
    Number of misclassifications in each epoch. 
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

# %% [markdown]
# ## Fitting the model
# 
# The model used to fit using the feature matrix X and the training data's true labels as a 1D column vector(?)
# 
# The `update` line uses the equation $\eta * (y - \hat{y})$, where $\hat{y}$ 
# 
# Q: What does the error rate as a 1D array represent?
# 

# %%
def fit(self, X, y):
        """
        Fit the model using training data. 
        
        :param self: A reference to this class. 
        :param X: The feature matrix of n observations and d features. 
        Shape: [n_examples, n_features]
        :param y: The *actual* class numbers (I think?)

        Returns
        -------
        self: A reference to this class.
        """

        # generate the random state of the object using the random_state var.
        rgen = np.random.RandomState(self.random_state)
        
        # randomize the weights to small values. 
        # X.shape[1] = number of features. Each feature has a weight. 
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size= X.shape[1])
        
        self.b = np.float_(0.)     # why the 0.?
        
        # tracks the misclassification rate after 
        self.errors = []

        # for the given number of epochs...
        for _ in range(self.n_iter):
            errors = 0                 # initialize error counter. 

            # create ordered pairs of input vector and actual result. 
            # repeat for all i observations. 
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))   # calculate the loss. 
                self.w_ += update * xi    # update the weight for this feature by the input. 
                self.b_ += update          # update the overall bias. 
                errors += int(update != 0.0)   #if there are classification errors, record it. 
            self.errors_.append(errors)

        return self                  

# %% [markdown]
# ### Other Utilites
# 
# TODO figure out what some of this means. 

# %%
def net_input(self, X):
    """
    Calculate the net input. 
    
    :param self: reference to self.
    :param X: Feature matrix. 
    """

    return np.dot(X, self.w_) + self.b_

def predict(self, X):
    """
    
    Given new data, try and predict based on the current model fit. 
    
    :param self: reference to this class
    :param X: Feature matrix. 
    """

    return np.where(self.net_input(X) >= 0.0, 1, 0)

# %% [markdown]
# ## TASK: Bias Absorption
# 
# Modify the above code to absorb the bias into the weights. 

# %%
# TODO: absorb bias into weights.

# def fit_absorb(self, X, y): 


