#!/usr/bin/env python3

import numpy as np
from math import e, log

xs = [[1,1], [0,0], [1,0], [0,1]]
ys = [0,0,1,0]
theta = np.array([1, 0]).T
theta_not = -0.5
h = []
sigmas = []
loss = 0

for i in range(4):
	x = np.array(xs[i]).T
	z = theta.T@x + theta_not
	h.append(z)
	sigma = 1/(1 + e**(-z))
	sigmas.append(sigma)
	loss += -(ys[i]*log(sigma) + (1 - ys[i])*log(1-sigma))
print(loss)
