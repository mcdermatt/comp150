from kalman import kalman
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

plt.figure()
# plt.xlim((-100,100))
# plt.ylim((-100,100))

A = np.array([[1,1],[0,1]])
B = np.array([[0.5],[1]])
C = np.array([[1,0]])
# process noise
w = 1
#measurement noise
v = 10
#v = np.array([[1e-4, 1e-5],[1e-4,1e-5]])

kal = kalman(A,B,C,dim=1,w=w,v=v)

u = 1 #input acceleration
post = np.array([[0],[0]])
x = np.array([[0],[0]])
curVar = np.array([[1,1],[1,1]])

count = 0
while count < 50:
	#randomly decide of gps should be checked
	p = np.random.rand()
	if p > 0.7:
		measTaken = 1
	else:
		measTaken = 0
	#run prediction step
	predRes = kal.prediction(x, post, curVar, u)
	x = predRes[0]
	#print('x = ', x)
	xhat = predRes[1]
	#print('xhat = ', xhat)
	predVar = predRes[2]
	plt.plot(x[0],x[1],'b.')
	# if measTaken == 1:
	# 	plt.plot(xhat[0],xhat[1],'go')
	plt.xlabel('pos')
	plt.ylabel('vel')
	#print(priori)
	#print('predVar =',predVar)

	#run update step
	updateRes = kal.update(x,xhat,predVar,measTaken)
	#print('post = ', updateRes[0], '   curVar = ', updateRes[1])
	post = updateRes[0]
	#plot estimates as red dots
	# plt.plot(post[0],post[1],'r.')
	if measTaken == 1:
		GPS, = plt.plot(updateRes[2],post[1],'g.')
	#plt.plot(post[0],post[1],'g.')
	curVar = updateRes[1]
	#print('curVar = ',curVar)

	#caculate ellipse
	#s = 0.203 #P(s<0.95) = 1-0.05 = 0.95 
	E, V = np.linalg.eig(curVar)
	lamx = E[1]
	lamy = E[0]
	#print('E = ', E)
	#print('V = ', V)
	n = np.argmax(np.abs(E)) ## arg max always resulting in 0,1
	#n = np.argmin(np.abs(E))
	a = V[:,n] #a = largest eigenvector
	#print('a = ', a)
	majorLen = 2*np.sqrt(lamx) #was 2*sqrt(s*lamx)
	print('majorLen = ', majorLen)
	minorLen = 2*np.sqrt(lamy)
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

	count += 1
	#print(count)
	plt.draw()
	plt.pause(0.1)
	
plt.pause(3)