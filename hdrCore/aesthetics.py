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
import copy, colour, skimage.transform, math, os
import sklearn.cluster, skimage.transform
import numpy as np
import functools
from . import processing, utils, image
import preferences.preferences as pref
from timeit import default_timer as timer



# -----------------------------------------------------------------------------
# --- Class ImageAestheticsModel ----------------------------------------------
# -----------------------------------------------------------------------------
class ImageAestheticsModel():
    """class ImageAestheticsModel: abstract class for image aesthetics model

        Static methods:
            build
    """
    def build(processPipe, **kwargs): return ImageAestheticsModel()

# -----------------------------------------------------------------------------
# --- Class Palette -----------------------------------------------------------
# -----------------------------------------------------------------------------
class Palette(ImageAestheticsModel):
    """class Palette(ImageAestheticsModel):  color palette

        Attributes:
            name (str): palette name
            colorSpace (colour.models.RGB_COLOURSPACES): colorspace): colorspace 
            nbColors (int):   number of colors in the Palette
            colors(numpy.ndarray): colors in the palette 
                        colors[0:nbColors,0:2]
                        sorted according to distance to black (in the palette colorSpace)
            type (image.imageType): image type (SDR|HDR)

        Methods:
            createImageOfPalette
            __repr__
            __str__

        Static methods:
            build
    """    
    # constructor
    def __init__(self, name, colors, colorSpace, type):
        """
        constructor of aesthetics.Palette:

        """
        self.name       = name
        self.colorSpace = colorSpace
        self.nbColors   = colors.shape[0]
        self.colors     = np.asarray(sorted(colors.tolist(), key = lambda u  : np.sqrt(np.dot(u,u))))
        self.type       = type

    @staticmethod    
    def build(processpipe, nbColors=5, method='kmean-Lab', processId=-1, **kwargs):
        """build: create the Palette from an image
        
            Args:
                processpipe (hdrCore.processing.ProcessPipe, Required): processpipe
                nbColors (int, Optionnal): number of colors in the palette (5 default values)
                method (str, Optionnal): 'kmean-Lab' (default value)
                processIdx (int, Optionnal): set the process after wihich computation of color palette is done
                default= -1 at the end of editing
                kwargs (dict, Otionnal): supplemental parameters according to method

            Returns:
                (hdrCore.aesthetics.Palette)
        """
        # get image according to processId
        image_ = processpipe.processNodes[processId].outputImage

        # according to method
        if method == 'kmean-Lab':
            # taking into acount supplemental parameters of 'kmean-Lab'
            #  'removeblack' : bool
            defaultParams = {'removeBlack': True}
            if 'removeBlack' in kwargs: removeBlack = kwargs['removeBlack']
            else: removeBlack = defaultParams['removeBlack']

            # get image according to processId
            image_ = processpipe.processNodes[processId].outputImage


            # to Lab then to Vector
            imageLab = processing.ColorSpaceTransform().compute(image_,dest='Lab')
            imgLabDataVector = utils.ndarray2vector(imageLab.colorData)

            if removeBlack:
                # k-means: nb cluster = nbColors + 1
                kmeans_cluster_Lab = sklearn.cluster.KMeans(n_clusters=nbColors+1)
                kmeans_cluster_Lab.fit(imgLabDataVector)

                cluster_centers_Lab = kmeans_cluster_Lab.cluster_centers_
                
                # remove darkness one
                idxLmin = np.argmin(cluster_centers_Lab[:,0])                           # idx of darkness
                cluster_centers_Lab = np.delete(cluster_centers_Lab, idxLmin, axis=0)   # remove min from cluster_centers_Lab

            else:
                # k-means: nb cluster = nbColors
                kmeans_cluster_Lab = sklearn.cluster.KMeans(n_clusters=nbColors)
                kmeans_cluster_Lab.fit(imgLabDataVector)
                cluster_centers_Lab = kmeans_cluster_Lab.cluster_centers_

            colors = cluster_centers_Lab
        else: colors = None

        return Palette('Palette_'+image_.name,colors, image.ColorSpace.Lab(), image_.type)

    def createImageOfPalette(self, colorWidth=100):
        """
        """
        if self.colorSpace.name =='Lab':
            if self.type == image.imageType.HDR :
                cRGB = processing.Lab_to_sRGB(self.colors, apply_cctf_encoding=True)
            else:
                cRGB = processing.Lab_to_sRGB(self.colors, apply_cctf_encoding=False)

        elif self.colorSpace.name=='sRGB':
            cRGB = self.colors
        width = colorWidth*cRGB.shape[0]
        height=colorWidth
        # return image
        img = np.ones((height,width,3))

        for i in range(cRGB.shape[0]):
            xMin= i*colorWidth
            xMax= xMin+colorWidth
            yMin=0
            yMax= colorWidth
            img[yMin:yMax, xMin:xMax,0]=cRGB[i,0]
            img[yMin:yMax, xMin:xMax,1]=cRGB[i,1]
            img[yMin:yMax, xMin:xMax,2]=cRGB[i,2]
        # colorData, name, type, linear, colorspace, scalingFactor
        return image.Image(
            '.',self.name,
            img,  
            image.imageType.SDR, False, image.ColorSpace.sRGB())

    # __repr__ and __str__
    def __repr__(self):
   
        res =   " Palette{ name:"           + self.name                 + "\n"  + \
                "          colorSpace: "    + self.colorSpace.name      + "\n"  + \
                "          nbColors: "      + str(self.nbColors)        + "\n"  + \
                "          colors: \n"      + str(self.colors)          + "\n " + \
                "          type: "          + str(self.type)            + "\n }"  
        return res

    def __str__(self) :  return self.__repr__()
# -----------------------------------------------------------------------------
# --- Class MultidimensionalImageAestheticsModel ------------------------------
# -----------------------------------------------------------------------------
class MultidimensionalImageAestheticsModel():
    """class MultidimensionalImageAestheticsModel contains Multiple Image Aesthetics Model:
            1 - color palette
            2 - composition convex hull 
            3 - composition strength lines
    
    """
    def __init__(self, processpipe):
        self.processpipe = processpipe
        self.processPipeChanged = True
        self.imageAestheticsModels = {}

    def add(self, key, imageAestheticsModel):
        self.imageAestheticsModels[key] = imageAestheticsModel

    def get(self, key):
        iam = None
        if key in self.imageAestheticsModels: iam = self.imageAestheticsModels[key]
        return iam

    def build(self, key, builder, processpipe):
        if not isintance(key,list):
            key, builder = [key],[builder]
        for k in key:
            self.add(k,builder.build(processpipe))
# -----------------------------------------------------------------------------
