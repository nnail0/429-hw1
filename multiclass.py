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

class Multiclass():
    """
    Docstring for MultiPerceptron
    """

    def __init__(self):
        self.setosa_neuron = Perceptron()
        self.versicolor_neuron = Perceptron()
        self.virginica_neuron = Perceptron()
            

def main():
    mc = Multiclass()
    mc.__init__()

    # read data from data file
    data = pd.read_csv("data/iris.data", header=None, encoding='utf-8') 
    # print(data.tail())

    # find feature matrix and true class labels from data
    X = data.iloc[:, [0, 1, 2, 3]].to_numpy()
    Y_ref = data.iloc[:, 4].to_numpy()

    # make copies and modify for each perceptron. 
    Y_setosa = Y_ref.copy()
    Y_versicolor = Y_ref.copy()
    Y_virginica = Y_ref.copy()

    Y_species = {"Iris-setosa" : Y_setosa, "Iris-virginica" : Y_virginica, "Iris-versicolor" : Y_versicolor}

    for name, arr in Y_species.items():
        for i in range(arr.size):
            if (arr[i] == name):
                arr[i] = 1
            else:
                arr[i] = 0
    
    mc.setosa_neuron.fit(X, Y_setosa)
    print("setosa trained")
    mc.versicolor_neuron.fit(X, Y_versicolor)
    print("versicolor trained")
    mc.virginica_neuron.fit(X, Y_virginica)
    print("virginica trained")

if __name__ == "__main__":
    main()

        

        
        


