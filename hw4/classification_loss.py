#!/usr/bin/env python3

from math import log, pi, e
import numpy as np
 
th = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1, 1]]).T
y = np.array([[0, 1, 0]]).T
 
 
def loss_nll(y, g):
	return -(y*np.log(g) + (1-y)@np.log(1-g))
	
def loss_nllm(y, g):
	return -y@np.log(g)
	
def derv_loss_nllm(x, y, g):
	return x@(g-y).T
	
def guess(x, theta):
	return theta.T@x
	
def softmax(x, theta):
	z = guess(x, theta)
	sum_ = np.sum(np.exp(z), axis = 0)
	
	try:
		softmax = np.exp(z)/sum_
	except:
		raise Exception("division by zero")
		
	return np.around(softmax, 3)
	
def gradient_update(x, y, theta, gradient, step, updates_num = 1):
	for _ in range(updates_num):
		theta = theta - step*gradient(x, y, softmax(x, theta))
	return np.around(theta, 3)
	
###Print derivative loss at x, y, theta
#print(derv_loss_nllm(x,y, softmax(x, th)))

###Print updated theta with step size, step, after updates_num steps.
step = 0.5
#print(gradient_update(x, y, th, derv_loss_nllm, step))

###Print softmax distribution after first step.
#print(softmax(x, gradient_update(x, y, th, derv_loss_nllm, step)))
