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
    TODO - Documentation de la classe quality
    
    Attributes:
        _image: TODO
            TODO
        imageNpath: TODO
            TODO
        user: TODO
            TODO
        score: TODO
            TODO
        artifact: TODO
            TODO
    """
    
    def __init__(self):
        """
        TODO - Documentation de la méthode __init__
        
        /!\ - Les constructeurs n'apparaissent pas dans la doc générées par sphinx.
        """
        self._image =       None
        self.imageNpath =    {'name':None, 'path': None}
        self.user =         {'pseudo': None}
        self.score =        {'quality': 0,'aesthetics':0, 'confort':0,'naturalness':0}
        self.artifact =     {'ghost':False, 'noise':False, 'blur':False, 'halo':False, 'other':False}

    def toDict(self):
        """
        Convert the object into a dict.
        
        Returns:
            TODO
                TODO
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
                TODO
        """
        return str(self.toDict())

    def __str__(self):
        """
        Convert to a string value.
        
        Returns:
            str
                TODO
        """
        return self.__repr__()
