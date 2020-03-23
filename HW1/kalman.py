import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

class kalman:
	"""creates kalman filter about specified params"""
	A = np.array([[1,1],[0.5,1]])
	B = np.array([[1],[0]])
	C = np.array([[1,0]])
	u = np.array([[0,0]])

	startPos = 0
	startVel = 0

	def __init__(self,A = A, B = B, C=C, dim = 1, w = 0, v = 0, u=u):
		self.dimension = dim #defaults to 1d case
		self.pnoise = w #procecss noise (wind etc.)
		self.mnoise = v #measurement noise (gps error etc)
		self.A = A
		self.B = B
		self.C = C
		#self.lastEst = 0
		#self.prevVar = 10
		#self.u = u
		self.Q = self.pnoise**2 * np.array([[0.25,0.5],[0.5, 1]])
		self.R = 0
		self.I = np.identity(dim+1)

		#Q # covariance matrix of noise in states
		#R # Covariance matrix of measurement noise

	def prediction(self, lastEst, prevVar, u):
		"""prediction step of kalman filter"""

		#predicted position for next step
		priori = np.dot(self.A,(lastEst)) + self.B*u + self.pnoise*self.B*np.random.randn() #self.pnoise*np.array([[0.5*np.random.randn()],[np.random.randn()]])

		#predicted variance for next step
		predVar = self.A.dot(prevVar*self.A.transpose()) + self.Q

		#prediction results
		predRes = [priori,predVar]
		return(predRes)

	def update(self, priori, predVar):
		"""update step of kalman filter"""
		#kalman gain 2x1 Matrix
		K = self.A.dot(predVar).dot(self.C.transpose())/(self.C.dot(predVar).dot(self.C.transpose()) + self.R)

		#posterior estimate
		y = np.array([1,0]).dot(priori) + self.mnoise*np.random.randn()
		post = priori + K*(y - self.C.dot(priori))

		#estimated current variance
		curVar = (self.I - K*self.C)*predVar

		#results of update
		updateRes = [post,curVar]
		return(updateRes)

	def display(self):
		"""shows values on plot"""
		#print(self.A)
		pass