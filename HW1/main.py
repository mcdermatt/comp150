from kalman import kalman
import numpy as np

A = np.array([[1,1],[0.5,1]])
B = np.array([[1],[0]])
C = np.array([1,0])

kal = kalman(A,B,dim=1)

kal.display()

u = np.array([[0,0]])
lastEst = np.array([[0],[0]]) #should be array(?)
prevVar = np.array([10,10]) #also not sure if this should be array or scalar

count = 0
while count < 1:

	#run prediction step
	predRes = kal.prediction(lastEst, prevVar, u)
	#print('priori = ',predRes[0], '  ','predVar = ', predRes[1])
	priori = predRes[0]
	predVar = predRes[1]
	print(priori,' ',predVar)

	#run update step
	updateRes = kal.update(priori,predVar)
	#print('post = ', updateRes[0], '   curVar = ', updateRes[1])
	post = updateRes[0]
	curVar = updateRes[1]

	#store current values as previous step values
	lastEst = post
	prevVar = curVar
	count += 1