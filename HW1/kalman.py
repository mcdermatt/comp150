import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

class kalman:
	"""creates kalman filter about specified params"""
	A = np.array([[1,1],[0.5,1]])
	B = np.array([[1],[0]])
	C = np.array([1,0])
	u = np.array([0,0])

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
		self.Q = np.identity(dim+1)
		self.R = np.identity(dim+1)
		self.I = np.identity(dim+1)

		#Q = covariance matrix of noise in states
		#R = Covariance matrix of measurement noise

	def prediction(self, lastEst, prevVar, u):
		"""prediction step of kalman filter"""

		#predicted position for next step
		priori = np.dot(self.A,(lastEst)) + np.dot(u,self.B) #ToDo - flip B and u
	
		#predicted variance for next step
		# predVar	= self.A*prevVar*self.A.transpose() #+ self.Q
		predVar = self.A.dot(prevVar).dot(self.A.transpose())
		
		#prediction results
		predRes = [priori,predVar]
		return(predRes)

	def update(self, priori, predVar):
		"""update step of kalman filter"""
		#kalman gain(?)
		#print(self.C.dot(predVar).dot(self.C.transpose())) #debug
		print(predVar.dot(self.C.transpose())) #this is resulting in a constant- 
			#do I need to transpose predVar??
		K = np.divide((predVar.dot(self.C.transpose())),(self.C.dot(predVar).dot(self.C.transpose()) ))#+self.R))
		print(K)

		#posterior estimate
		y = np.array([1,0])*priori
		post = priori + K*(y - self.C*priori)
		
		#estimated current variance
		curVar = (self.I - K*self.C)*predVar

		#results of update
		updateRes = np.array([post,curVar])
		return(updateRes)

	def display(self):
		"""shows values on plot"""
		#print(self.A)
		pass