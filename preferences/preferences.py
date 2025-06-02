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
    """load preferences file: prefs.json

            Args:

            Returns (Dict)
        
    """
    with open('./preferences/prefs.json') as f: return  json.load(f)
# -----------------------------------------------------------------------------
def savePref():
    global HDRdisplays
    global HDRdisplay
    global imagePath
    pUpdate = {
            "HDRdisplays" : HDRdisplays,
            "HDRdisplay"  : HDRdisplay,
            "imagePath"   : imagePath
        }
    if verbose: print(" [PREF] >> savePref(",pUpdate,")")
    with open('./preferences/prefs.json', "w") as f: json.dump(pUpdate,f)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# loading pref
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
print("uHDRv6: loading preferences")
p = loadPref()
if p :
    HDRdisplays = p["HDRdisplays"]
    HDRdisplay = p["HDRdisplay"]
    imagePath = p["imagePath"]
else:
    HDRdisplays = {
        'none' :                {'shape':(2160,3840), 'scaling':1,   'post':'',                          'tag': "none"},
        'vesaDisplayHDR1000' :  {'shape':(2160,3840), 'scaling':12,  'post':'_vesa_DISPLAY_HDR_1000',    'tag':'vesaDisplayHDR1000'},
        'vesaDisplayHDR400' :   {'shape':(2160,3840), 'scaling':4.8, 'post':'_vesa_DISPLAY_HDR_400',     'tag':'vesaDisplayHDR400'},
        'HLG1' :                {'shape':(2160,3840), 'scaling':1,   'post':'_HLG_1',                    'tag':'HLG1'}
        }
    # current display
    HDRdisplay = 'vesaDisplayHDR1000'
    imagePath = '.'
print(f"       target display: {HDRdisplay}")
print(f"       image path: {imagePath}")
# -----------------------------------------------------------------------------
# --- Functions computation ---------------------------------------------------
# -----------------------------------------------------------------------------
def getComputationMode():
    """returns the preference computation mode: python, numba, cuda, ...

        Args:

        Returns (str)
    """
    return computation
# -----------------------------------------------------------------------------
# --- Functions HDR dispaly ---------------------------------------------------
# -----------------------------------------------------------------------------
def getHDRdisplays():
    """returns the current display model

    Args:

    Returns (Dict)
    """
    return HDRdisplays
# -----------------------------------------------------------------------------
def getHDRdisplay():
    """returns the current display model

    Args:

    Returns (Dict)
    """
    return HDRdisplays[HDRdisplay]
# -----------------------------------------------------------------------------
def setHDRdisplay(tag):
    """set the HDR display

        Args:
            tag (str): tag of HDR display, must be a key of HDRdisplays

        Returns:
    """
    global HDRdisplay
    if tag in HDRdisplays: HDRdisplay =tag
    savePref()
# ----------------------------------------------------------------------------
def getDisplayScaling():  return getHDRdisplay()['scaling']
# ----------------------------------------------------------------------------
def getDisplayShape():  return getHDRdisplay()['shape']
# -----------------------------------------------------------------------------
# --- Functions path ---------------------------------------------------
# -----------------------------------------------------------------------------
def getImagePath(): return imagePath if os.path.isdir(imagePath) else '.'
# ----------------------------------------------------------------------------
def setImagePath(path): 
    global imagePath
    imagePath = path
    if verbose: print(" [PREF] >> setImagePath(",path,"):",imagePath)
    savePref()
# ----------------------------------------------------------------------------
             