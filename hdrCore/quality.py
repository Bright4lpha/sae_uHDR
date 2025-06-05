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
import json, os, copy
import numpy as np
from . import utils, processing, image
# -----------------------------------------------------------------------------
# --- Class quality ----------------------------------------------------------
# -----------------------------------------------------------------------------
class quality(object):
    """
    Class representing the quality of an image.
    This class contains attributes for image information, user details, scores, and artifacts.
    Attributes:
        _image (hdrCore.image.Image): The image object.
        imageNpath (dict): Dictionary containing the name and path of the image.
        user (dict): Dictionary containing user information, such as pseudo.
        score (dict): Dictionary containing quality scores for various aspects.
        artifact (dict): Dictionary indicating the presence of artifacts in the image.
    Methods:
        toDict(): Converts the object into a dictionary representation.
        __repr__(): Returns a string representation of the object.
        __str__(): Returns a string representation of the object.
    
    """
    
    def __init__(self):
        """
        Initialize the quality object with default values.
        
        /!\ - Les constructeurs n'apparaissent pas dans la doc générées par sphinx.
        """
        self._image =       None
        self.imageNpath =    {'name':None, 'path': None}
        self.user =         {'pseudo': None}
        self.score =        {'quality': 0,'aesthetics':0, 'confort':0,'naturalness':0}
        self.artifact =     {'ghost':False, 'noise':False, 'blur':False, 'halo':False, 'other':False}

    def toDict(self):
        """
        Convert the quality object into a dictionary representation.
        This method returns a dictionary containing the image name and path, user information, quality scores, and artifacts.
        Returns:
            dict: A dictionary representation of the quality object.
        """
        return {'image': copy.deepcopy(self.imageNpath),
                              'user':copy.deepcopy(self.user),
                              'score':copy.deepcopy(self.score),
                              'artifact':copy.deepcopy(self.artifact)}

    def __repr__(self):
        """
        Convert to a string value.
        
        Returns:
            str
                A string representation of the quality object, formatted as a JSON string.
        """
        return str(self.toDict())

    def __str__(self):
        """
        Convert to a string value.
        
        Returns:
            str
                A string representation of the quality object, formatted as a JSON string.
        """
        return self.__repr__()
