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
# --- Package preferences -----------------------------------------------------
# -----------------------------------------------------------------------------
"""
package preferences contains all global variables that stup the preferences.

"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
# RCZT 2023
# import numba, json, os, copy
import numpy as np, json, os

# -----------------------------------------------------------------------------
# --- Preferences -------------------------------------------------------------
# -----------------------------------------------------------------------------
target = ['python','numba','cuda']
computation = target[0]
# verbose mode: print function call 
#   usefull for debug
verbose = True
# list of HDR display takien into account
#   red from prefs.json file
#   display info:
#   "vesaDisplayHDR1000":                           << display tag name
#       {
#           "shape": [ 2160, 3840 ],                << display shape (4K)
#           "scaling": 12,                          << color space scaling to max
#           "post": "_vesa_DISPLAY_HDR_1000",       << postfix add when exporting file
#           "tag": "vesaDisplayHDR1000"             << tag name
#       }
HDRdisplays = None
# current HDR display: tag name in above list
HDRdisplay = None
# image size when editing image: 
#   small size = quick computation, no memory issues
maxWorking = 1200
# last image directory path
imagePath ="."
# keep all metadata
keepAllMeta = False
# -----------------------------------------------------------------------------
# --- Functions preferences --------------------------------------------------
# -----------------------------------------------------------------------------
def loadPref():
    """
    Load preferences file: prefs.json

    Args:
        None

    Returns:
        dict: The loaded preferences.
    """
    with open('./preferences/prefs.json') as f:
        return json.load(f)

def savePref():
    """
    Save preferences to the preferences file: prefs.json

    Args:
        None

    Returns:
        None
    """
    global HDRdisplays
    global HDRdisplay
    global imagePath
    pUpdate = {
        "HDRdisplays": HDRdisplays,
        "HDRdisplay": HDRdisplay,
        "imagePath": imagePath
    }
    if verbose:
        print(" [PREF] >> savePref(", pUpdate, ")")
    with open('./preferences/prefs.json', "w") as f:
        json.dump(pUpdate, f)

# Loading preferences
print("uHDRv6: loading preferences")
p = loadPref()
if p:
    HDRdisplays = p["HDRdisplays"]
    HDRdisplay = p["HDRdisplay"]
    imagePath = p["imagePath"]
else:
    HDRdisplays = {
        'none': {'shape': (2160, 3840), 'scaling': 1, 'post': '', 'tag': "none"},
        'vesaDisplayHDR1000': {'shape': (2160, 3840), 'scaling': 12, 'post': '_vesa_DISPLAY_HDR_1000', 'tag': 'vesaDisplayHDR1000'},
        'vesaDisplayHDR400': {'shape': (2160, 3840), 'scaling': 4.8, 'post': '_vesa_DISPLAY_HDR_400', 'tag': 'vesaDisplayHDR400'},
        'HLG1': {'shape': (2160, 3840), 'scaling': 1, 'post': '_HLG_1', 'tag': 'HLG1'}
    }
    # Current display
    HDRdisplay = 'vesaDisplayHDR1000'
    imagePath = '.'
print(f"       target display: {HDRdisplay}")
print(f"       image path: {imagePath}")

def getComputationMode():
    """
    Returns the preference computation mode: python, numba, cuda, etc.

    Args:
        None

    Returns:
        str: The computation mode.
    """
    return computation

def getHDRdisplays():
    """
    Returns the current display models.

    Args:
        None

    Returns:
        dict: The current display models.
    """
    return HDRdisplays

def getHDRdisplay():
    """
    Returns the current display model.

    Args:
        None

    Returns:
        dict: The current display model.
    """
    return HDRdisplays[HDRdisplay]

def setHDRdisplay(tag):
    """
    Set the HDR display.

    Args:
        tag (str): Tag of HDR display, must be a key of HDRdisplays.

    Returns:
        None
    """
    global HDRdisplay
    if tag in HDRdisplays:
        HDRdisplay = tag
    savePref()

def getDisplayScaling():
    """
    Returns the scaling factor of the current display.

    Args:
        None

    Returns:
        float: The scaling factor of the current display.
    """
    return getHDRdisplay()['scaling']

def getDisplayShape():
    """
    Returns the shape of the current display.

    Args:
        None

    Returns:
        tuple: The shape of the current display.
    """
    return getHDRdisplay()['shape']

def getImagePath():
    """
    Returns the current image path.

    Args:
        None

    Returns:
        str: The current image path.
    """
    return imagePath if os.path.isdir(imagePath) else '.'

def setImagePath(path):
    """
    Set the image path.

    Args:
        path (str): The path to set.

    Returns:
        None
    """
    global imagePath
    imagePath = path
    if verbose:
        print(" [PREF] >> setImagePath(", path, "):", imagePath)
    savePref()

# ----------------------------------------------------------------------------
             