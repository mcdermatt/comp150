import numpy as np
import cv2
from matplotlib import pyplot as plt

#number of initial particles in each direction
fidelity = 10
droneFOV = 300 #length of sides of square image taken by drone camera

#Main Map
plt.figure()
img = cv2.imread('MarioMap.png',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

particlePosArr = np.zeros([3,fidelity**2]) #particle number, xpos, ypos

#draw initial particles
x = np.linspace(0,img.shape[0],fidelity)
y = np.linspace(0,img.shape[1],fidelity)
particleCount = 0
for i in x:
	for j in y:
		particlePosArr[:,particleCount] = [particleCount,i,j]
		plt.plot(i,j,'ro')
		particleCount += 1
print(particlePosArr)


#get random initial position for drone
dronex = int(np.floor(np.random.rand()*img.shape[0]))
droney = int(np.floor(np.random.rand()*img.shape[1]))
#plot image taken by drone camera
plt.figure()
dronePic = img[dronex:dronex+droneFOV,droney:droney+droneFOV]
#get negative of drone camera image
#dronePic = cv2.bitwise_not(dronePic)


plt.imshow(dronePic,cmap = 'gray', interpolation = 'bicubic')


#Cropped Map
plt.figure()
#loop through particle positions drawing cropped map in figure 2
for k in particlePosArr[0]:
	imgCropped = img[int(particlePosArr[1,int(k)-1]):int(particlePosArr[1,int(k)-1]+droneFOV),int(particlePosArr[2,int(k)-1]):int(particlePosArr[2,int(k)-1]+droneFOV)]
	plt.imshow(imgCropped, cmap = 'gray', interpolation = 'bicubic')

	#compare cropped image to inverse drone image (find difference)

	#record closest matching particle numbers

	plt.draw()
	plt.pause(0.05)
	plt.clf()

#generate new particles

#movement


plt.show()

