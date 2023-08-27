#!/usr/bin/env python3

import numpy as np

#def lin_reg_predict(x, th, th0):
#        return th.T@x + th0

def lin_reg_predict(X, th, th0):
        predictions = th.T@X + th0

        return predictions

#def mse(Y, Y_hat):
#        return np.array([[np.mean((Y - Y_hat)**2)]])
    
def mse(Y, Y_hat):
        return np.mean((Y - Y_hat)**2, axis = 1, keepdims = True)

def lin_reg_err(X, Y, th, th0):
        predictions = lin_reg_predict(X, th, th0)
        return mse(predictions, Y)

def random_regress(X, Y, k):
	d, n = X.shape
	thetas = 2 * np.random.rand(d, k) - 1
	th0s = 2 * np.random.rand(1, k) - 1
	errors = lin_reg_err(X, Y, thetas, th0s.T)
	i = np.argmin(errors)
	theta, th0 = thetas[:,[i]], th0s[:,[i]]
	return (theta, th0), errors[i]

my_X = np.array([[1,2,3,4]])
my_Y = np.array([[2,7,-3,1]])
my_k = 4        
print(random_regress(my_X, my_Y, my_k))        

