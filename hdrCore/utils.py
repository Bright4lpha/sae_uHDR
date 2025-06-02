# uHDR: HDR image editing software
#   Copyright (C) 2021  remi cozot 
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
# hdrCore project 2020
# author: remi.cozot@univ-littoral.fr

# -----------------------------------------------------------------------------
# --- Package hdrCore ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
package hdrCore consists of the core classes for HDR imaging.

"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import os, math
import numpy as np

# -----------------------------------------------------------------------------
# --- Package functions -------------------------------------------------------
# -----------------------------------------------------------------------------
def filenamesplit(filename):
    """retrieve path, name and extension from a filename.

    Args:
        filename (str,Required): filename
            
    Returns:
        (str,str,str): (path,name,ext)
            
    Example:
        filenamesplit("./dir0/dir1/name.ext") returns ("./dir0/dir1/","name","ext")
    """
    
    path, nameWithExt = os.path.split(filename)
    splits = nameWithExt.split('.')
    ext = splits[-1].lower()
    name = '.'.join(splits[:-1])
    return (path,name,ext)

def filterlistdir(path,extList):
    """return the files in path that end by one of the string in extList

    Args:
        path (str, Required): path to directory
        extList (tuple of str, Required)
            
    Returns:
        ([str])or (tuple of str) or (str): list, tuple or single str
           
    Example:
        filterlistdir('./images/', ('.jpg', '.JPG', '.png'))
    """ 
    
    ext = None
    if isinstance(extList, list):
        ext = tuple(extList)
    elif isinstance(extList, str):
        ext = extList
    elif isinstance(extList, tuple):
        ext = extList
    filenames = sorted(os.listdir(path))
    res = list(filter(lambda x: x.endswith(ext),filenames))

    return res

def ndarray2vector(nda):
    """transform 2D array of color data to vector

    Args:
        nda (numpy.ndarray, Required): numpy array (2D of scalar or vector)
            
    Returns:
        (numpy.ndarray, Required): numpy array (1D of scalar or vector)
    """
    if len(nda.shape) ==2 :
        x,y = nda.shape
        c = 1
    elif len(nda.shape) ==3:
        x,y,c = nda.shape
    return np.reshape(nda, (x * y, c))


# ------------------------------------------------------------------------------------------
#def linearWeightMask(x, xMin, xMax, xTolerance):
#    if x < (xMin - xTolerance):     y = 0
#    elif x <= xMin:                 y = (x -(xMin - xTolerance))/xTolerance
#    elif x <= xMax :                y = 1
#    elif x <= (xMax + xTolerance):  y = 1 - (x - xMax)/xTolerance
#    else:                           y = 0
#    return y

# ------------------------------------------------------------------------------------------

def NPlinearWeightMask(x,xMin,xMax,xTolerance):
    """
    TODO - Documentation de la méthode NPlinearWeightMask

    Args:
        x: TODO
            TODO
        xMin: TODO
            TODO
        xMax: TODO
            TODO
        xTolerance: TODO
            TODO
            
    Returns:
        TODO
            TODO
    """
    # reshape x
    h,w  = x.shape  # 2D array
    xv = np.reshape(x,(h*w,1))
    y = np.ones((h*w,1))
    y = np.where((xv <= (xMin - xTolerance)),               0,y)                                        # (0)                +-----------+
    y = np.where((xv > (xMin - xTolerance))&(xv <= xMin),   (xv -(xMin - xTolerance))/xTolerance,y)     # (1)               /             \
    y = np.where((xv > (xMin))&(xv <= xMax),                1,y)                                        # (2)              /               \
    y = np.where((xv > (xMax))&(xv <= xMax + xTolerance),   1 - (xv - xMax)/xTolerance,y)               # (3)     --------+                 +-------
    y = np.where((xv > (xMax + xTolerance)),                0,y)                                        # (4)         (0)   (1)  (2)    (3)    (4)

    return np.reshape(y,(h,w))

def croppRotated(h,w,alpha):
    """
    TODO - Documentation de la méthode croppRotated

    Args:
        h: TODO
            TODO
        w: TODO
            TODO
        alpha: TODO
            TODO
            
    Returns:
        TODO
            TODO
    """

    cosA = math.cos(math.radians(alpha))
    sinA = math.sin(math.radians(alpha))

    v_up_left = - h/(h*cosA + w*sinA)
    v_up_right = h/(h*cosA - w*sinA)

    v = min(abs(v_up_left),abs(v_up_right))

    return (h*v, w*v)

# ------------------------------------------------------------------------------------------
# ---- Constants ---------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
#######################
#HDRdisplay = {
#    'none' :                {'scaling':1,  'post':'',                       'tag':None},
#    'vesaDisplayHDR1000' :  {'scaling':12, 'post':'_vesa_DISPLAY_HDR_1000', 'tag':'vesaDisplayHDR1000'}
#    }
# ------------------------------------------------------------------------------------------
