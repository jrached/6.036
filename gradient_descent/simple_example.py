#!/usr/bin/env python3 

theta_1 = 3
theta_2 = -2

func = -3*theta_1 - theta_1*theta_2 + 2*theta_2 + theta_1**2 + theta_2**2 
#print(func)


def gradient_descent():
	points = [(1,1), (1, -1), (1,1), (1, -1)]
	n = 1
	theta = -1/9
	thetas = []
	for x, y in points:
		theta = theta - 0.2*(2/n)*(x*theta - y)*x
		thetas.append(theta)
	return thetas

test_thetas = gradient_descent()
print(test_thetas)
