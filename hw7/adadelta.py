#!/usr/bin/env python3

import sys 
from math import sqrt

x = float(sys.argv[1])
grad = float(sys.argv[2])

def adadelta(init_val, gradient, gamma = 0.5, n = 0.1, err = 0.0001, steps = 3):
	g = gradient
	G = g**2
	W = init_val
	Ws = []
	
	for _ in range(steps):
		G = gamma*G + (1.0-gamma)*g**2
		W = W - n*g/sqrt(G+err)
		Ws.append(W)
	
	print(Ws)

adadelta(x, grad)

