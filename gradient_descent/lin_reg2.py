#!/usr/bin/env python3

import numpy as np

def minimizing_theta(X, Y):
	return np.linalg.inv(X@X.T)@X@Y.T
	
def ridge_minimizing_theta(X,Y):
	#TODO 
	pass
	
def mse(X, Y, Theta):
	n = len(X[0,:])
	return (X.T@Theta-Y.T).T@(X.T@Theta-Y.T)/n
	
	
my_X = np.array([[1,2,3,4],[0,0,0,0], [0,0,0,0], [1,1,1,1]])
my_Y = np.array([[2,7,-3,1]])	
#my_theta = minimizing_theta(my_X, my_Y)
my_theta = np.array([[4.70697, -1.28107*10**6, 1.10581*10**7, -1.03437*10**2]]).T
print(mse(my_X, my_Y, my_theta))
