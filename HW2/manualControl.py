import numpy as np
import cv2
from matplotlib import pyplot as plt

#number of initial particles in each direction
fidelity = 20
droneFOV = 70 #length of sides of square image taken by drone camera
particleNoise = 5
movementNoise = 15
speed = 15
runtime = 30

#Main Map
plt.figure(0)
img = cv2.imread('MarioMap.png',0) #load in image

#reduce image size
scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

particlePosArr = np.zeros([3,fidelity**2]) #particle number, xpos, ypos

#draw initial particles
x = np.linspace(0,img.shape[0]-droneFOV,fidelity)
y = np.linspace(0,img.shape[1]-droneFOV,fidelity)
particleCount = 0
for i in x:
	for j in y:
		particlePosArr[:,particleCount] = [particleCount,i,j]
		plt.plot(i,j,'r.')
		particleCount += 1

#get random initial position for drone
dronex = int(np.floor(np.random.rand()*(img.shape[0] - droneFOV)))  
droney = int(np.floor(np.random.rand()*(img.shape[1] - droneFOV)))
print("Actual Coords of Drone: ",dronex, " ", droney)

runNum = 1
while runNum <= runtime:
	#plot image taken by drone camera
	plt.figure(1)
	plt.clf()
	dronePic = img[int(dronex):int(dronex+droneFOV),int(droney):int(droney+droneFOV)]
	#get negative of drone camera image
	#dronePic = cv2.bitwise_not(dronePic)
	plt.imshow(dronePic,cmap = 'gray', interpolation = 'bicubic')
	plt.pause(0.05)
	plt.draw()

	#loop through particle positions drawing cropped map in figure 2
	worstFit = 255*(fidelity**2)
	fitness = np.zeros([fidelity**2])
	cumFitness = np.zeros([fidelity**2])
	bestFit = 0
	bestPt = 0

	for k in particlePosArr[0]:
		#Cropped Map
		# plt.figure(2)
		imgCropped = img[int(particlePosArr[1,int(k)-1]):int(particlePosArr[1,int(k)-1]+droneFOV),int(particlePosArr[2,int(k)-1]):int(particlePosArr[2,int(k)-1]+droneFOV)]
		# plt.imshow(imgCropped, cmap = 'gray', interpolation = 'bicubic')

		#record closest matching particle numbers
		# plt.draw()
		# plt.pause(0.05)
		# plt.clf()
		if (imgCropped.shape == dronePic.shape):
			#compare cropped image to inverse drone image (find difference)
			overlap = np.sum(np.subtract(imgCropped, dronePic))
			fitness[int(k)-1] = worstFit - overlap
		else:
			#set arbitrarily low value - if you set to zero point will never be looked at again
			fitness[int(k)-1] = 0

		if fitness[int(k)-1] > bestFit:
			bestFit = fitness[int(k)-1]
			bestPt = int(k)
			print("Best Match is ", int(k))

		#main map
		#plt.figure(0)
		#display where current search point is

#	print(fitness)
	#weigh fitness of points for resampling
	sumOfFitness = np.sum(fitness)
	#print("sum of fitness: ", sumOfFitness)
	cumFitness[0] = fitness[0]
	m = 0
	while m < fitness.shape[0]:
		fitness[m] = fitness[m]/sumOfFitness
		cumFitness[m] = fitness[m] + cumFitness[m-1]
		m += 1

	#pick new points to resample based on cumFitness
	n = 0
	while n < particlePosArr.shape[1]:
		r = np.random.rand() #random variable
		#check to see which value in cumFitness got picked
		p = 0
		while p < fitness.shape[0]:
			if cumFitness[p] > r:
				#update particlePosArray with value closes to randomly chosen point + noise
				particlePosArr[1,n] = particlePosArr[1,p] + np.floor(particleNoise*np.random.randn())
				particlePosArr[2,n] = particlePosArr[2,p] + np.floor(particleNoise*np.random.randn())
				break
			p += 1
		n += 1


	#print(particlePosArr)


	#update map with particles
	plt.figure(0)
	plt.clf()
	plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	# plt.xlim((img.shape[1],0))
	# plt.ylim((img.shape[0],0))
	# u = 0
	# while u < particlePosArr.shape[1]:
	# 	plt.plot(particlePosArr[2,u],particlePosArr[1,u],'b.')
	# 	u+=1
	#plt.plot(particlePosArr[1,:],particlePosArr[2,:],'b.')
	#plt.pause(0.05)
	#plt.draw()

	print("most likely coords of drone: ", particlePosArr[:,bestPt])
	#draw most likely estimated location on main map
	#plt.plot(particlePosArr[2,bestPt],particlePosArr[1,bestPt],'go')
	#plt.pause(0.5)
	#plt.draw()

	#movement
	theta = 2*np.pi*np.random.rand()
	#print("theta = ", theta)
	dx = speed*np.cos(theta)
	dy = speed*np.sin(theta)
	#don't go outside the map
	if (dronex + dx > img.shape[0] - droneFOV) or (droney + dy > img.shape[0] - droneFOV):
		dx = 0
		dy = 0
	#update drone position
	dronex = dronex + dx
	droney = droney + dy
	#update pos array
	particlePosArr[1,:] = particlePosArr[1,:] + dx + movementNoise
	particlePosArr[2,:] = particlePosArr[2,:] + dy + movementNoise
	z, = plt.plot(particlePosArr[1,:],particlePosArr[2,:],'g.')
	plt.draw()
	plt.pause(0.05)
	z.remove()
	runNum += 1

plt.show()

