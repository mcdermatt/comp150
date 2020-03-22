import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

class kalman:
	"""creates kalman filter about specified params"""
	A = np.array([1,1])
	B = np.array([1,0])
	C = np.array([1,0])
	startPos = 0
	startVel = 0

	def __init__(self,A = A, B = B, C=C, dim = 1, w = 0, v = 0, u=0):
		self.dimension = dim #defaults to 1d case
		self.pnoise = w #procecss noise (wind etc.)
		self.mnoise = v #measurement noise (gps error etc)
		self.A = A
		self.B = B
		self.C = C
		#self.lastEst = 0
		#self.prevVar = 10
		#self.u = u
		self.Q = 0
		self.I = np.identity(dim+1)

		#Q = covariance matrix of noise in states
		#R = Covariance matrix of measurement noise

	def prediction(self, lastEst, prevVar, u):
		"""prediction step of kalman filter"""

		#predicted position for next step
		priori = self.A*lastEst + self.B*u

		#predicted variance for next step
		predVar	= self.A*prevVar*(self.A.transpose()) + self.Q

		return(priori,predVar)

	def update(self, priori, predVar):
		"""update step of kalman filter"""
		#kalman gain(?)
		K = (predVar * self.C.transpose())/(self.C*predVar*self.C.transpose()+self.R)

		#posterior estimate
		y = np.array([1,0])*x
		post = priori + K*(y - self.C*priori)
		
		#estimated current variance
		curVar = (self.I - K*self.C)*predVar

		return(post, curVar)

	def display(self):
		"""shows values on plot"""
		print(self.A)
