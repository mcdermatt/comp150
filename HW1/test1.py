import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver
r_std, q_std = 2., 0.003
dt = 0.1
cv = KalmanFilter(dim_x=2, dim_z=1)
cv.x = np.array([[0., 1.]]) # position, velocity
cv.F = np.array([[1, dt],[0, 1]])
cv.R = np.array([[r_std**2]])
cv.H = np.array([[1., 0.]])
cv.P = np.diag([.1**2, .03**2])
cv.Q = Q_discrete_white_noise(2, dt, q_std**2)
saver = Saver(cv)
for z in range(100):
    cv.predict()
    cv.update([z + randn() * r_std])
    saver.save() # save the filter's state
saver.to_array()
plt.plot(saver.x[:, 0])
# plot all of the priors
plt.plot(saver.x_prior[:, 0])
# plot mahalanobis distance
plt.figure()
plt.plot(saver.mahalanobis)