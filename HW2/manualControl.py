import numpy as np
import cv2
from matplotlib import pyplot as plt

#number of initial particles in each direction
fidelity = 15
droneFOV = 70 #length of sides of square image taken by drone camera

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
#plot image taken by drone camera
plt.figure(1)
dronePic = img[dronex:dronex+droneFOV,droney:droney+droneFOV]
#get negative of drone camera image
#dronePic = cv2.bitwise_not(dronePic)


plt.imshow(dronePic,cmap = 'gray', interpolation = 'bicubic')



#loop through particle positions drawing cropped map in figure 2
fitness = np.zeros([fidelity**2])
bestFit = 10e7
bestPt = 0

for k in particlePosArr[0]:
	#Cropped Map
	plt.figure(2)
	imgCropped = img[int(particlePosArr[1,int(k)-1]):int(particlePosArr[1,int(k)-1]+droneFOV),int(particlePosArr[2,int(k)-1]):int(particlePosArr[2,int(k)-1]+droneFOV)]
	plt.imshow(imgCropped, cmap = 'gray', interpolation = 'bicubic')

	#record closest matching particle numbers
	plt.draw()
	plt.pause(0.05)
	plt.clf()
	if (imgCropped.shape == dronePic.shape):
		#compare cropped image to inverse drone image (find difference)
		overlap = np.sum(np.subtract(imgCropped, dronePic))
		fitness[int(k)-1] = overlap
	else:
		#set arbitrarily low value for now
		fitness[int(k)-1] = 10e7

	if fitness[int(k)-1] < bestFit:
		bestFit = fitness[int(k)-1]
		bestPt = int(k)
		print("Best Match is ", int(k))

	#main map
	#plt.figure(0)
	#display where current search point is


print(fitness)

print("most likely coords of drone: ", particlePosArr[:,bestPt])
#draw most likely estimated location on main map
plt.figure(0)
plt.plot(particlePosArr[2,bestPt],particlePosArr[1,bestPt],'go')
plt.pause(0.05)
plt.draw()

#generate new particles

#movement


plt.show()

