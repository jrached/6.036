#!/usr/bin/env python3

import numpy as np

#####################
###BASIC FUNCTIONS###
#####################
###Hypothesis, sigmoid, loss, objective function.

#returns vector of same shape as y
def hypothesis(X, y, th, th0):
        return th.T@X + np.full(y.shape, th0)

# returns a vector of the same shape as z
def sigmoid(z):
    return 1/(np.exp(-z)+1)

# X is dxn, y is 1xn, th is dx1, th0 is 1x1
# returns a (1,n) array for the nll loss for each data point given th and th0
def nll_loss(X, y, th, th0):
    g = sigmoid(th.T@X + np.full((th.T@X).shape, th0))
    return -(y*np.log(g) + (1-y)*np.log(1-g))

# X is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
# returns a scalar for the llc objective over the dataset
def llc_obj(X, y, th, th0, lam):
    n = X.shape[1]
    loss = nll_loss(X, y, th, th0)
    return ((1/n)*np.sum(loss) + lam*th.T@th)[0,0]

##########################
###DERIVATIVE FUNCTIONS###
##########################

def d_sigmoid(z):
    sig = sigmoid(z)
    return sig - sig**2

# returns a (d,n) array for the gradient of nll_loss(X, y, th, th0) with respect to th for each data point
def d_nll_loss_th(X, y, th, th0):
    z = hypothesis(X, y, th, th0)
    g = sigmoid(z)
    return -X*((1.0-g)*y - g*(1.0-y))
    
# returns a (1,n) array for the gradient of nll_loss(X, y, th, th0) with respect to th0
def d_nll_loss_th0(X, y, th, th0):
    z = hypothesis(X, y, th, th0)
    g = sigmoid(z)
    return -((1.0-g)*y - g*(1.0-y))
    
# returns a (d,1) array for the gradient of llc_obj(X, y, th, th0) with respect to th
def d_llc_obj_th(X, y, th, th0, lam):
    d, n = X.shape
    return np.sum(d_nll_loss_th(X, y, th, th0), axis = 1, keepdims = True)/n + 2*lam*th

# returns a (1,1) array for the gradient of llc_obj(X, y, th, th0) with respect to th0
def d_llc_obj_th0(X, y, th, th0, lam):
    d, n = X.shape
    return np.array([[np.sum(d_nll_loss_th0(X,y, th, th0))/n]])

# returns a (d+1, 1) array for the full gradient as a single vector (which includes both th, th0)
def llc_obj_grad(X, y, th, th0, lam):
    return np.vstack((d_llc_obj_th(X, y, th, th0, lam), d_llc_obj_th0(X, y, th, th0, lam)))


#################
###GD FUNCTION###
#################
#Basic gradient descent procedure.

def gd(f, df, x0, step_size_fn, num_steps):
    fx = f(x0)
    dfx = df(x0)
    x = x0
    
    for n in range(num_steps):
        print(x.shape)
        print(dfx.shape)
        x = x - step_size_fn(n)*dfx
        fx = f(x)
        dfx = df(x)
        
    return x, fx

###################
###MAIN FUNCTION###
###################
#Performs gradient descent using the derivative of the objective.

def llc_min(data, labels, lam):
    """
    Parameters:
        data: dxn
        labels: 1xn
        lam: scalar
    Returns:
        same output as gd
    """
    num_steps = 10
    d, n = data.shape
    x0 = np.zeros((d+1, 1)) 
    
    def llc_min_step_size_fn(i):
       return 2/(i+1)**0.5
    
    def obj(thetas):
        th, th0 = thetas[:d,:], thetas[d, 0]
        return llc_obj(data, labels, th, th0, lam)
    
    def d_obj(thetas):
        th, th0 = thetas[:d,:], thetas[d,0]
        return llc_obj_grad(data, labels, th, th0, lam)
        
    return gd(obj, d_obj, x0, llc_min_step_size_fn, num_steps)



################
###TEST CASES###
################

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

# Test case 1
x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator
ans = llc_obj(x_1, y_1, th1, th1_0, .1)

# Test case 2
ans = llc_obj(x_1, y_1, th1, th1_0, 0.0)

###Main function test case
def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, 0, 1, 0]])
    return X, y
sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

x_1, y_1 = super_simple_separable()
ans = package_ans(llc_min(x_1, y_1, 0.0001))

x_1, y_1 = separable_medium()
ans = package_ans(llc_min(x_1, y_1, 0.0001))

#print(ans)
