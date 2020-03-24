from kalman import kalman
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

plt.figure()
# plt.xlim((-40,40))
# plt.ylim((-40,40))

A = np.array([[1,1],[0,1]])
B = np.array([[0.5],[1]])
C = np.array([[1,0]])
# process noise
w = 1
#measurement noise
v = 10
#v = np.array([[1e-4, 1e-5],[1e-4,1e-5]])

kal = kalman(A,B,C,dim=1,w=w,v=v)

u = 0 #input acceleration
lastEst = np.array([[0],[0]])
prevVar = np.array([[1,1],[1,1]])

usum = 0

count = 0
while count < 100:

	#run prediction step
	predRes = kal.prediction(lastEst, prevVar, u)
	priori = predRes[0]
	predVar = predRes[1]
	plt.plot(priori[0],priori[1],'r.')
	plt.xlabel('pos')
	plt.ylabel('vel')
	#print(priori)
	#print('predVar =',predVar)

	#run update step
	updateRes = kal.update(priori,predVar)
	#print('post = ', updateRes[0], '   curVar = ', updateRes[1])
	post = updateRes[0]
	curVar = updateRes[1]
	print('curVar = ',curVar)

	#caculate ellipse
	s = 0.203 #P(s<0.95) = 1-0.05 = 0.95 
	E, V = np.linalg.eig(curVar)
	lamx = E[1]
	lamy = E[0]
	print('E = ', E)
	print('V = ', V)
	n = np.argmax(np.abs(E)) ## arg max always resulting in 0,1
	#n = np.argmin(np.abs(E))
	a = V[:,n] #a = largest eigenvector
	print('a = ', a)
	majorLen = 2*np.sqrt(s*lamx)
	print('majorLen = ', majorLen)
	minorLen = 2*np.sqrt(s*lamy)
	print('minorLen = ', minorLen)

	t_rotGPS = np.arctan(a[0]) + np.pi/2
	print('t_rot = ',t_rotGPS)
	n = np.linspace(0, 2*np.pi, 100)
	EllGPS = np.array([majorLen*np.cos(n) , minorLen*np.sin(n)])
	#rotation matrix
	R_rotGPS = np.array([[np.cos(t_rotGPS) , -np.sin(t_rotGPS)],[np.sin(t_rotGPS) , np.cos(t_rotGPS)]])
	
	Ell_rotGPS = np.zeros((2,EllGPS.shape[1]))
	for i in range(EllGPS.shape[1]):
		Ell_rotGPS[:,i] = np.dot(R_rotGPS,EllGPS[:,i])

	#plot rotated elipse (centered around (0,0))
	try:
		elipseRotGPS.remove()
	except:
		pass
	elipseRotGPS, = plt.plot( post[0]+Ell_rotGPS[0,:] , post[1]+Ell_rotGPS[1,:],'b--' )


	#store current values as previous step values
	lastEst = post
	prevVar = curVar
	count += 1
	#print(count)
	plt.draw()
	plt.pause(0.001)
	
plt.pause(10)