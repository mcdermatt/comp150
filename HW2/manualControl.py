import numpy as np
import cv2
from matplotlib import pyplot as plt

#number of initial particles in each direction
fidelity = 10

#import image taken from drone camera

#get negative of drone camera image

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

#Cropped Map
plt.figure()
#loop through particle positions drawing cropped map in figure 2
for k in particlePosArr[1]:
	imgCropped = img[100:150,300:350]
	plt.imshow(imgCropped, cmap = 'gray', interpolation = 'bicubic')

	#compare cropped image to inverse drone image (find difference)

	#record closest matching particle numbers

#generate new particles

#movement


plt.show()

