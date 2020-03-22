from kalman import kalman
import numpy as np

A = np.array([6,9])
B = np.array([9,6])

kal = kalman(A,B,dim=1)

kal.display()

u = 0
lastEst = 0
prevVar = 10

count = 0
while count < 10:

	priori, predVar = kal.prediction(lastEst, prevVar, u)
	print('priori = ',priori, '  ','predVar = ', predVar)

	lastEst = post
	prevVar = curVar

	post, curVar = kal.update(priori,predVar)
	print('post = ', post, '   curVar = ', curVar)

	count += 1