import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
k_average = [] #average accuracy for k's
time_avg = [] #avearga time for k's
np.set_printoptions(suppress=True)
def Bestk(X, Y, maxk):
    k_accuracy = []  #split the data
    k_time = []
    for i in range(1,maxk + 1):
        for j in range(0,50):
            starttime = time.time()
            X_train, x_test, y_train, y_test = train_test_split(X,Y)
            knn_i = KNeighborsRegressor(n_neighbors = i)
            knn_i.fit(X_train, y_train) #training the model
            k_i_accuracy = knn_i.score(x_test,y_test)
            endtime = time.time() - starttime
            k_accuracy.append(k_i_accuracy)
            k_time.append(endtime)
        k_average.append(np.average(k_accuracy))
        time_avg.append(np.average(k_time))
        k_accuracy = []
        k_time = []
    bestk = np.argmax(k_average) + 1
    return bestk 

def Summary(accuracy, times,maxk):
    sum_matrix = np.zeros((maxk,3))
    sum_matrix[:, 0] = [k for k in range(1, maxk + 1)]
    sum_matrix[:, 1] = k_average
    sum_matrix[:, 2] = time_avg
    return sum_matrix

def Summary_graphs(accuracy, times, maxk):
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,14))
    axes[0, 0].plot([k for k in range(1, maxk + 1)], k_average)
    axes[0, 1].plot([k for k in range(1,maxk + 1)], time_avg)
    axes[0, 0].set_title("Average accuracy for each k value")
    axes[0, 1].set_title("Average time for each k value")
    axes[0, 0].set_xlabel("max K's")
    axes[0, 1].set_xlabel("max K's")
    axes[0, 0].set_ylabel("Average Accuracy")
    axes[0, 1].set_ylabel("Average Time")
 
iris = pd.read_csv("C:\Users\Ricardo\Desktop\datasets_19_420_Iris.csv")
iris.head()
X = iris.iloc[:, 1:5].values
Y = iris.iloc[:, :5].values
k = Bestk(X, Y, 6)
summary = Summary(k_average,time_avg,6)
graphs = Summary_graphs(k_average,time_avg,6)
print(graphs)
"""
print(k_average)
print(time_avg)
print(k) 
"""   