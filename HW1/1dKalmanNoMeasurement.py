import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

#Part 1: assumes no measurement

plt.figure()
plt.xlim((-20,20))
plt.ylim((-20,20))
plt.xlabel("Position")
plt.ylabel("Velocity")
sigma = 1
x = 0
dx = 0
ddx = 0
u = 1 #control vector
runlen = 5
stdGPS = 2 #standard deviation of GPS measurements
GPSfail = False
lastVel = 0
lastGPS = 0
lastT = 0
oldVariance = 15
oldMean = 0

R = (sigma**2)*np.array([[0.5],[1]])*np.array([0.5,1]) #not sure if right
print(R)

#position and velocity
X = np.array([[x],[dx]])
t=0
while t < runlen:

	#prediction matrix, if timestep !=1,  A = np.array([[1, timestep],[0, 1]])
	A = np.array([[1, 1],[0, 1]])

	ddx = np.random.randn()
	B = np.array([[0.5*ddx],[1*ddx]])

	print("ddx = ",ddx)

	#epsilon = N([[0],[0]],R)
	#eps = np.random.randn(2,1)*R
	#print(eps)

	#Enviornmental uncertainty (add in with each step)
	#95th percentile = 2 standard deviations
	Q = 2

	Xprev = X	
	#covariance matrix 
	R = A*R*A.transpose() + Q#*1.01**t
	#print(R)
	eig = np.linalg.eig(R)
	eig = eig[0] #only take first eigenvalue
	#print(eig)

	#print(R)

	#draw boat position
	boatpos, = plt.plot(X[0],X[1],'ro')
	#prediction of position  ------ X = A*X + B*u
	X = A.dot(X) + B.dot(u) #+ eps
	print(X)

	#draw elipse of uncertainty for next boat position
	#rotation angle based on covariance matrix
	t_rot = float(np.arctan(R[0,1]/R[1,0]))
	n = np.linspace(0, 2*np.pi, 100)

	Ell = np.array([R[1,1]*np.cos(n) , 0.5*R[0,0]*np.sin(n)])
	#Ell = np.array([2*np.sqrt(eig[1])*np.cos(n) , 2*np.sqrt(eig[0])*np.sin(n)])
	#rotation matrix
	R_rot = np.array([[np.cos(t_rot) , -np.sin(t_rot)],[np.sin(t_rot) , np.cos(t_rot)]])
	
	Ell_rot = np.zeros((2,Ell.shape[1]))
	for i in range(Ell.shape[1]):
		Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

	#plot rotated elipse (centered around (0,0))
	elipseRot, = plt.plot( 0+Ell_rot[0,:] , 0+Ell_rot[1,:],'b' )
	#plt.show()
	plt.pause(0.001)
	plt.draw()
	t += 1
plt.pause(5)