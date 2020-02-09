import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

#Part 1: assumes no measurement

plt.figure()
plt.xlim((-50,50))
plt.ylim((-20,20))
sigma = 1
x = 0
dx = 0
ddx = 0
u = 1 #control vector
runlen = 20

#position and velocity
X = np.array([[x],[dx]])
t=0
while t < runlen:

	#prediction matrix, if timestep !=1,  A = np.array([[1, timestep],[0, 1]])
	A = np.array([[1, 1],[0, 1]])

	ddx = np.random.randn()
	B = np.array([[0.5*ddx],[1*ddx]])

	#noise covariance matrix
	R = (sigma**2)*np.array([[0.5],[1]])*np.array([0.5,1]) #not sure if right
	#print(R)

	#epsilon = N([[0],[0]],R)
	#eps = np.random.randn(2,1)*R
	#print(eps)

	#Enviornmental uncertainty
	Q = 5

	Xprev = X	
	#uncertainty 
	R = A*R*A.transpose() + Q

	#draw boat position
	boatpos, = plt.plot(X[0],X[1],'ro')
	#prediction of position  ------ X = A*X + B*u
	X = A.dot(X) + B.dot(u) #+ eps

	#draw elipse of uncertainty for next boat position
	#t_rot = np.arctan(2/1) #rotation angle
	t_rot = float(np.arctan((Xprev[1]-X[1])/(Xprev[0]-X[0])))
	n = np.linspace(0, 2*np.pi, 100)

	Ell = np.array([R[1,0]*np.cos(n) , R[0,1]*np.sin(n)])
	#rotation matrix
	R_rot = np.array([[np.cos(t_rot) , -np.sin(t_rot)],[np.sin(t_rot) , np.cos(t_rot)]])
	
	Ell_rot = np.zeros((2,Ell.shape[1]))
	for i in range(Ell.shape[1]):
		Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

	elipseRot, = plt.plot( X[0]+Ell_rot[0,:] , X[1]+Ell_rot[1,:],'b' )
	#plt.show()
	plt.pause(0.125)
	plt.draw()
	elipseRot.remove()
	t += 1