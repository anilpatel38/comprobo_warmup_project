import statistics
import random
import math


def ransac(x,y):
    """Finds most robust line.

    Args:
        x (list): list of integers that represent x values of points.
        y (lists): list of integers that represent y values of points.
    Returns:
        None if line is parallel or there is no line, slope if not.
    """
    n = len(x)
    threshold = .2
    xrange = abs(max(x)-min(x))
    yrange = abs(max(y)-min(y))
    threshold = threshold * statistics.mean([xrange, yrange])/20
    final_slope = None
    maxcount = 0
    for i in range(n):
        random_indexes = random.choices(list(range(n)), k=2)
        i1 = random_indexes[0]
        i2 = random_indexes[1]

        x1 = x[i1]
        y1 = y[i1]

        x2 = x[i2]
        y2 = y[i2]
        run = x2 - x1
        if run == 0:
            slope = None
        else:
            slope = (y2 - y1) / run
            b = y1 - slope*x1

        count = 0;

        for m in range(1,n):
            p0 = (x[m-1],y[m-1])
            p = (x[m], y[m])
            dist_points = math.sqrt(math.pow((x[m]-x[m-1]),2)+math.pow((y[m]-y[m-1]),2))
            new_y = slope * x[m-1] + b
            dif_y = abs(new_y - y[m-1])

            if dist_points <= .15 and dif_y <= .05:
                count += 1
            elif dist_points > .15 and count > 0:
                break
        print(slope, count)
        if count > maxcount:
            maxcount = count
            final_slope = slope

    return final_slope
