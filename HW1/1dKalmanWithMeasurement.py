import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

#Part 1: assumes no measurement

plt.figure()
# plt.xlim((-50,50))
plt.ylim((-10,10))
plt.xlabel("Position")
plt.ylabel("Velocity")
sigma = 1
x = 0
dx = 0
ddx = 0
u = 1 #control vector
runlen = 50
stdGPS = 2 #standard deviation of GPS measurements
GPSfail = False
lastVel = 0
lastGPS = 0
lastT = 0
oldVariance = 100
oldMean = 0

R = (sigma**2)*np.array([[0.5],[1]])*np.array([0.5,1]) #not sure if right
print(R)

#position and velocity
X = np.array([[x],[dx]])
t=0
while t < runlen:
	#get gps measurement
	if t % 5 == 0:
		#add sensor noise
		z = X[0] + stdGPS*np.random.randn()
		print("gps measurement: ", z)
		#calculate velocity from GPS
		vel = (z - lastGPS)/(t-lastT)
		lastT = t
		lastGPS = z
		#plot gps point on map
		gpsPt = plt.plot(z,vel,'gx')

		#combine measurements
		mean = (stdGPS**2/(stdGPS**2+oldVariance))*z + (oldVariance**2/(stdGPS**2+oldVariance**2))*oldMean 
		oldMean = mean
		variance = np.sqrt(((stdGPS**2/(stdGPS**2+oldVariance))**2)*stdGPS*stdGPS + ((oldVariance**2/(stdGPS**2+oldVariance**2))**2)*oldVariance)
		oldVariance = variance

		#plot new elipse representing CI accounting for gps
		#rotation angle based on covariance matrix
		t_rotGPS = float(np.arctan(R[0,1]/R[1,0]))
		n = np.linspace(0, 2*np.pi, 100)

		EllGPS = np.array([10*np.cos(n) , 5*np.sin(n)])
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
		elipseRotGPS, = plt.plot( mean+Ell_rotGPS[0,:] , vel+Ell_rotGPS[1,:],'g' )

	#prediction matrix, if timestep !=1,  A = np.array([[1, timestep],[0, 1]])
	A = np.array([[1, 1],[0, 1]])

	ddx = np.random.randn()
	B = np.array([[0.5*ddx],[1*ddx]])

	#epsilon = N([[0],[0]],R)
	#eps = np.random.randn(2,1)*R
	#print(eps)

	#Enviornmental uncertainty (add in with each step)
	#95th percentile = 2 standard deviations
	Q = 2

	Xprev = X	
	#covariance matrix 
	R = A*R*A.transpose() + Q#*1.01**t
	print(R)
	eig = np.linalg.eig(R)
	eig = eig[0] #only take first eigenvalue
	#print(eig)

	#print(R)

	#draw boat position
	boatpos, = plt.plot(X[0],X[1],'ro')
	#prediction of position  ------ X = A*X + B*u
	X = A.dot(X) + B.dot(u) #+ eps
	#print(X[1])

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
	try: 
		elipseRot.remove()
	except:
		pass
	elipseRot, = plt.plot( 0+Ell_rot[0,:] , 0+Ell_rot[1,:],'b' )
	#plt.show()
	plt.pause(0.0025)
	plt.draw()
	t += 1
plt.pause(5)