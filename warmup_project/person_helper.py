from numpy import arctan
import math

def get_points_ahead(x, y, total_angle):
    x_list = x[0:int(total_angle/2)]+x[int(360-total_angle/2):360]
    y_list = y[0:int(total_angle/2)]+y[int(360-total_angle/2):360]
    return x_list, y_list

def center_of_mass(x, y):
    x_cord = sum(x)/len(x)
    y_cord = sum(y)/len(y)
    return (x_cord, y_cord)

def cart_to_polar(x, y):
    r = math.sqrt(y**2 + x**2)
    print(math.sqrt(2))
    print(r)
    theta = numpy.arctan(y/x)
    return (r, theta)
