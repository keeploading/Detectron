import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import pylab as pl

obs = np.genfromtxt("../kalman_x.txt", delimiter=str(','))
pred = np.genfromtxt("../kalman_x_pred.txt", delimiter=str(','))
time = np.genfromtxt("../kalman_x_time.txt", delimiter=str(','))
est = np.genfromtxt("../kalman_x_est.txt", delimiter=str(','))
obs = obs[0:-1]
pred = pred[0:-1]
est = est[0:-1]
meter_scale = (3.5/100)
# b = b * meter_scale
time = time[0:-1]
obs_log = np.vstack((time, obs)).T
pred_log = np.vstack((time, pred)).T
est_log = np.vstack((time, est)).T
point_log_obs = ""
point_log_pred = ""
point_log_est = ""
for a_item in obs_log:
    point_log_obs += "({},{}),".format(a_item[0], a_item[1])
for a_item in pred_log:
    point_log_pred += "({},{}),".format(a_item[0], a_item[1])
for a_item in est_log:
    point_log_est += "({},{}),".format(a_item[0], a_item[1])

pl.figure()
observe = pl.plot(obs_log[:,0], obs_log[:,1], color='b')
predict = pl.plot(pred_log[:, 0], pred_log[:, 1], color='g')
filter = pl.plot(est_log[:, 0], est_log[:, 1], color='r')
# pl.show()
pl.legend((observe[0], predict[0], filter[0]),
          ('obs', 'predict', 'filt'),
          loc='lower left'
          )
pl.show()
