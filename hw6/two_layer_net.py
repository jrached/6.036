#!/usr/bin/env python 3

import numpy as np

X = np.array([[-2, 0, 2]])
Y = np.array([[0, 1, 0]])

ones = np.full((1, 3), 1)

X = np.vstack((X, ones))

w1 = np.array([[1], [-1.0]])
w2 = np.array([[-1], [-1.0]])

w = np.hstack((w1, w2))

# 
	
z = w.T@X
#fz = np.heaviside(z, 1)
print(z)



#fz = np.vstack((fz, ones))
#w = np.array([[1], [1], [-1.5]])
	
#fz = np.heaviside(z, 1)
#print(fz) 



