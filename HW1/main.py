from kalman import kalman
import numpy as np

A = np.array([[1,1],[0,1]])
B = np.array([[0.5],[1]])
C = np.array([[1,0]])
# process noise
w = 0.1
#measurement noise
v = 1
#v = np.array([[1e-4, 1e-5],[1e-4,1e-5]])

kal = kalman(A,B,C,dim=1,w=w,v=v)

kal.display()

u = 0 #input acceleration
lastEst = np.array([[0],[0]])
prevVar = np.array([[10,1],[1,10]])

count = 0
while count < 10:
	u = u + np.random.randn()
	print(u)

	#run prediction step
	predRes = kal.prediction(lastEst, prevVar, u)
	#print('priori = ',predRes[0], '  ','predVar = ', predRes[1])
	priori = predRes[0]
	predVar = predRes[1]
	#print(priori)
	#print('predVar =',predVar)

	#run update step
	updateRes = kal.update(priori,predVar)
	#print('post = ', updateRes[0], '   curVar = ', updateRes[1])
	post = updateRes[0]
	curVar = updateRes[1]
	print(post)
	#print(curVar)

	#store current values as previous step values
	lastEst = post
	prevVar = curVar
	count += 1
	print(count)