from matplotlib import pyplot as plt
import numpy as np

plt.figure()

x = 0
dx = 0
ddx = 0
X = [x,dx,ddx]
t = 0
runlen = 100

plt.hlines(0,-20,20)  # Draw a horizontal line
plt.yticks([])
plt.draw()

while t < runlen:
	#random change in acceleration fom -1 to 1
	ddx = np.random.randn(1)
	dx = dx + ddx
	x = x + dx + 0.5*ddx
	print(x,dx,ddx)
	boatpos, = plt.plot(x,0,'ro')
	#plt.eventplot(a, orientation='horizontal', colors='b') # plots vertical lines
	plt.pause(0.05)
	boatpos.remove()
	plt.draw()
	t += 1