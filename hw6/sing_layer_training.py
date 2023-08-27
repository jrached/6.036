#!/usr/bin/env python 3

import numpy as np

#Returns vector of shape (1,n) from inputs X, (d, n), and w, (1, d).
def compute_z(X, w): 
	d, n = X.shape
	ones = np.full((1, n), 1) 
	X = np.vstack((X, ones))
	return w.T@X

#Returns vector of same shape as z.
def compute_fz(z):
	return z if z > 0 else 0
	
#Returns loss vector of same shape as y (1, n).
def loss(g, y):
	return (g - y)**2 
	
#Computes derivative with respect to w. Returns vector of shape (2, n)
def d_loss(X, g, y):
	return 2*X@(g-y)

def d_loss_0(X, g, y):
	return 2*(g-y)
	
def loss_gradient(X, g, y):
	return np.vstack((d_loss(X, g, y), d_loss_0(X, g, y)))
	
#Return scalar 
def objective(g, y):
	return (1/n)*np.sum(loss(g, y))
	
def d_objective(g, y):
	return (1/n)*np.sum(d_loss(g, y))
	

current_x = np.array([[-1, 1]]).T
current_w = np.array([[5, 5, 1]]).T
current_y = np.array([[0]])
steps = 2

current_x = np.array([[-2, 0, 2]])
current_y = np.array([[0, 1, 0]])

def current_step(index):
	return 0.25 
	
def gradient_descent(X, y, w, step_func, num_steps):
	
	for i in range(num_steps):
		w = w - step_func(i)*loss_gradient(X, compute_fz(compute_z(X,w)), y)
		
		
	return (w, compute_fz(compute_z(X, w)))


#print(compute_fz(compute_z(current_x, current_w)))
#print(loss(compute_fz(compute_z(current_x, current_w)), current_y))

#Print gradient.
#print(d_loss(current_x, compute_fz(compute_z(current_x, current_w)), current_y))
#print(d_loss_0(current_x, compute_fz(compute_z(current_x, current_w)), current_y))

new_w, new_fz = gradient_descent(current_x, current_y, current_w, current_step, steps)

print(new_w)

#Print new gradient.
#print(d_loss(current_x, compute_fz(compute_z(current_x, new_w)), current_y))
#print(d_loss_0(current_x, compute_fz(compute_z(current_x, new_w)), current_y))


