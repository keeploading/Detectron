
from scipy.misc import comb
from scipy import optimize
import numpy as np
import math

def parabola2(x, A, B, C):
    return A*x*x + B*x + C

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i



def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000


        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return np.vstack((xvals, yvals)).T

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) **2 + (y1 - y2) **2)

def line_param(x1, y1, x2, y2):
    a = (y1 - y2) /(x1 - x2)
    return [a, y1 - a*x1]