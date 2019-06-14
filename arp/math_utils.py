
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

def get_parabols_by_points(points):
    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    if C > 960 or C < -960:
        print ("C > 960 or C < -960, please check:" + str(points))
    return [A, B, C]

def get_parabola_by_distance(coefficient, distance):
    A = coefficient[0]
    B = coefficient[1]
    C = coefficient[2]

    if A == 0:
        theta = math.atan2(B, 1)
        return [A, B, C + distance / math.sin(theta)]
    else:
        point1 = [-B/(2*A), (4*A*C - B*B)/(4*A) + distance]

        x_array = [-B/(2*A) + 100, -B/(2*A) + 200]
        source_p1 = [x_array[0], A * x_array[0] * x_array[0] + B * x_array[0] + C]
        source_p2 = [x_array[1], A * x_array[1] * x_array[1] + B * x_array[1] + C]

        theta = math.atan2(2*A*source_p1[0] + B, 1)
        point2 = [source_p1[0] - math.sin(theta) * distance, source_p1[1] + math.cos(theta) * distance]

        theta = math.atan2(2*A*source_p2[0] + B, 1)
        point3 = [source_p2[0] - math.sin(theta) * distance, source_p2[1] + math.cos(theta) * distance]
        #y=ax+b
        if (point3[0] * (point2[1] - point1[1]) + point2[0] * (point1[1] - point3[1]) + point1[0] * (point3[1] - point2[1])) == 0:
            return [0, 0, C + distance]
        return get_parabols_by_points([point1, point2, point3])