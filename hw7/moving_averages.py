#!/usr/bin/env python3 

import sys
import numpy as np

def one_d_momentum():
	a = [1,1,1,10,10]
	A = 0
	As = [A]
	gamma = float(sys.argv[1])

	for elem in a:
		A = gamma*A + (1-gamma)*elem
		As.append(A)
		
	print(As)
	
def two_d_momemtum():
	a = np.array([[1,1], [1, -1], [1, 1], [1, -1]])
	gamma = float(sys.argv[1])
	n = 1
	n_prime = n/(1-gamma)
	B = 0
	Bs = []
	
	for elem in a:
		B = gamma*B + (1-gamma)*elem
		
	
	print((-n_prime*B).tolist())
	
two_d_momemtum()
