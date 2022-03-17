# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import math
import numpy as np


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def euler2mat(angle):

    x, y, z = angle[0], angle[1], angle[2]
    cosz = math.cos(z)
    sinz = math.sin(z)

    zmat = np.array([cosz, -sinz, 0,sinz,  cosz, 0, 0,  0,  1]).reshape(3,3)

    cosy = math.cos(y)
    siny = math.sin(y)

    ymat = np.array([cosy, 0, siny, 0,  1, 0, -siny, 0,  cosy]).reshape(3,3)

    cosx = math.cos(x)
    sinx = math.sin(x)

    xmat = np.array([1, 0, 0, 0, cosx, -sinx, 0,  sinx,  cosx]).reshape(3,3)

    rotMat =np.matmul(np.matmul(zmat,ymat),xmat).astype(np.float32)

    return rotMat
