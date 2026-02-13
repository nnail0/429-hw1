# p4: minibatch_SGD

# %% [markdown]
# # Logistic Regression
# Nathan Nail - CS 429 Fall 2026
# 
# **NOTE** this model also uses gradient descent. 



# %%
import numpy as np
import pandas as pd
import os

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
        self.b_ = np.float64(0.)
        self.losses = []
    
        n_batches = int(np.ceil(X.shape[0] / self.batch_size))

        # list containing n_batches number of mini-batches. 
        num_lists = [[] for i in range(n_batches)]
        batch_Xs = np.ndarray((n_batches, self.batch_size, X.shape[1]))
        batch_Ys = np.ndarray((n_batches, self.batch_size))

    

        for i in range(self.n_iter):

            num_losses = 0

            # create a shuffled list of indices. 
            idx_list = np.arange(X.shape[0])
            np.random.shuffle(idx_list)

            # make the mini_batches by pulling from X and Y. 
            for i in range(self.batch_size - 1):                                # repeat _batchsize_ number of times
                for list_num in range(n_batches - 1):                          # for each list:
                    np.append(batch_Xs[list_num], X[idx_list[list_num + i]])     # place a value from X into a minibatch.
                    np.append(batch_Ys[list_num], y[idx_list[list_num + i]])

            for i in range(n_batches):
                net_input = self.net_input(batch_Xs[i])
                output = self.activation(net_input)
                # adjust sizing to batch size. 
                errors = batch_Ys[i] - output
                self.w_ += self.eta * 2.0 * batch_Xs[i].T.dot(errors)
                self.b_ += self.eta * 2.0 * errors.mean()
                loss = (-batch_Ys[i].dot(np.log(output)) - ((1-batch_Ys[i]).dot(np.log(1-output)))) / batch_Xs.shape[0]
                num_losses += loss
                
            self.losses.append(loss)

        # TODO add in per-epoch error tracking. 

        print(self.losses)
        return self
# %%
    def net_input(self, X):
        """
        Calculate the net input.
        """
        return np.dot(X, self.w_) + self.b_


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


def main():
    minibatchSGD = LogisticRegressionGD(0.01, 50, 1, 32)
    iris_data = pd.read_csv("data/iris.data")

    print(iris_data.tail())

    X = iris_data.iloc[:, 0:3].to_numpy()
    y = iris_data.iloc[:, 4].to_numpy()

    for i in range(len(y)):
        if (y[i] == "Iris-setosa"):
            y[i] = int(0)
        elif (y[i] == "Iris-versicolor"):
            y[i] = int(1)
        else:  # virginica
            y[i] = int(2)

    print(X[146:150])
    print(y[146:150])

    minibatchSGD.fit_mini_batch_sgd(X, y)
        

if __name__ == "__main__":
    main()


