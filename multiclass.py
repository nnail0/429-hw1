# 3. multiclass.py

import os 
import pandas as pd
import numpy as np
from perceptron import Perceptron

"""
Docstring for multiclass

This implementation uses a one-vs-many approach, where different perceptrons are used. 
- One perceptron per iris class
- Modify true outputs such that the correct species is 1 and the two incorrect species are 0. 
- 

"""

data = pd.read_csv("data/iris.data", header=None, encoding='utf-8')
# print(data.tail())

X = data.iloc[:, 0:3].to_numpy

print("X: ", X)

Y_ref = data.iloc[:, 4].to_numpy

print("Y_ref:")
print(Y_ref)

Y_setosa = np.copy(Y_ref)
print(Y_setosa.shape)
Y_versicolor = np.copy(Y_ref)
Y_virginica = np.copy(Y_ref)

Y_species = {"Iris_setosa" : Y_setosa, "Iris_virginica" : Y_virginica, "Iris_versicolor" : Y_versicolor}

for name, arr in Y_species.items():
    print(type(arr))
    for i in range(arr.size):
        arr[i] = np.where(i == name, 1, 0)

print("modified Y values:")
print(Y_setosa)
print(Y_versicolor)
print(Y_virginica)

# TODO find a way to modify the true class labels. 
setosa_neuron = Perceptron(eta=0.01, n_iter = 50, random_state = 1)
versicolor_neuron = Perceptron(eta = 0.01, n_iter=50, random_state = 1)
virginica_neuron = Perceptron(eta = 0.01, n_iter = 50, random_state = 1)

