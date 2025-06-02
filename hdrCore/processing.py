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
import pathos.multiprocessing, multiprocessing, subprocess
import numpy as np
import skimage.transform
import functools
from geomdl import BSpline
from geomdl import utilities
from . import image, utils, aesthetics
# RCZT 2023
# from . import image, utils, numbafun, aesthetics
import guiQt.controller as gc
import preferences.preferences as pref
from timeit import default_timer as timer

import hdrCore.coreC

# -----------------------------------------------------------------------------
# --- package functions -------------------------------------------------------
# -----------------------------------------------------------------------------
def XYZ_to_sRGB(XYZ,apply_cctf_encoding=True):
    """convert pixel array from XYZ to sRGB colorspace.

    Args:
        XYZ (numpy.ndarray, Required): array of pixels in XYZ colorspace.
        apply_cctf_encoding (boolean, Optionnal): True to encode with sRGB cctf encoding function.
            
    Returns:
        (numpy.ndarray): array of pixels in sRGB colorspace.
    """
    return  colour.XYZ_to_sRGB(XYZ, 
                               illuminant=np.array([ 0.3127, 0.329 ]), 
                               chromatic_adaptation_transform='CAT02', 
                               apply_cctf_encoding=apply_cctf_encoding)
  
def sRGB_to_XYZ(RGB,apply_cctf_decoding=True):
    """convert pixel array from sRGB to XYZ colorspace.

    Args:
        RGB (numpy.ndarray, Required): array of pixels in sRGB colorspace.
        apply_cctf_decoding (boolean, Optionnal): True to encode with sRGB cctf decoding function.
            
    Returns:
        (numpy.ndarray): array of pixels in XYZ colorspace.
    """
    return colour.sRGB_to_XYZ(RGB, 
                              illuminant=np.array([ 0.3127, 0.329 ]), 
                              chromatic_adaptation_transform='CAT02', 
                              apply_cctf_decoding=apply_cctf_decoding)           

def Lab_to_XYZ(Lab):
    """convert pixel array from Lab to XYZ colorspace.

    Args:
        Lab (numpy.ndarray, Required): array of pixels in Lab colorspace.
            
    Returns:
        (numpy.ndarray): array of pixels in XYZ colorspace.
    """
    return colour.Lab_to_XYZ(Lab, illuminant=np.array([ 0.3127, 0.329 ]))

def XYZ_to_Lab(XYZ):
    """convert pixel array from Lab to Lab colorspace.

    Args:
        XYZ (numpy.ndarray, Required): array of pixels in Lab colorspace.
            
    Returns:
        (numpy.ndarray): array of pixels in Lab colorspace.
    """
    return colour.XYZ_to_Lab(XYZ, illuminant=np.array([ 0.3127, 0.329 ]))

def Lab_to_sRGB(Lab, apply_cctf_encoding=True, clip = False):
    """convert pixel array from Lab to sRGB colorspace.

    Args:
        Lab (numpy.ndarray, Required): array of pixels in Lab colorspace.
        apply_cctf_encoding (boolean, Optionnal): True to encode with sRGB cctf encoding function.
        clip (boolean, Optionnal): False do not clip values beyond colorspace limits (RGB values < 0 or RGB values > 1).
            
    Returns:
        (numpy.ndarray): array of pixels in sRGB colorspace.
    """
    XYZ = colour.Lab_to_XYZ(Lab, illuminant=np.array([ 0.3127, 0.329 ]))
    RGB =  colour.XYZ_to_sRGB(XYZ,
                              illuminant=np.array([ 0.3127, 0.329 ]), 
                              chromatic_adaptation_transform='CAT02', 
                              apply_cctf_encoding=apply_cctf_encoding)
    if clip:
        RGB[RGB<0] = 0
        RGB[RGB>1] = 1
    return RGB

def sRGB_to_Lab(RGB, apply_cctf_decoding=True):
    """convert pixel array from sRGB to Lab colorspace.

    Args:
        RGB (numpy.ndarray, Required): array of pixels in RGB colorspace.
        apply_cctf_decoding (boolean, Optionnal): True to encode with sRGB cctf decoding function.
            
    Returns:
        (numpy.ndarray): array of pixels in sRGB colorspace.


    """
    XYZ = colour.sRGB_to_XYZ(RGB, 
                             illuminant=np.array([ 0.3127, 0.329 ]), 
                             chromatic_adaptation_transform='CAT02', 
                             apply_cctf_decoding=apply_cctf_decoding)           
    Lab = colour.XYZ_to_Lab(XYZ, illuminant=np.array([ 0.3127, 0.329 ]))
    return Lab

def Lch_to_sRGB(Lch,apply_cctf_encoding=True, clip=False):
    """convert pixel array from Lch to sRGB colorspace.

    Args:
        Lch (numpy.ndarray, Required): array of pixels in Lab colorspace.
        apply_cctf_encoding (boolean, Optionnal): True to encode with sRGB cctf encoding function.
        clip (boolean, Optionnal): False do not clip values beyond colorspace limits (RGB values < 0 or RGB values > 1).
            
    Returns:
        (numpy.ndarray): array of pixels in sRGB colorspace.
    """
    Lab = colour.LCHab_to_Lab(Lch)
    XYZ = colour.Lab_to_XYZ(Lab, illuminant=np.array([ 0.3127, 0.329 ]))
    RGB = colour.XYZ_to_sRGB(
        XYZ, illuminant=np.array([ 0.3127, 0.329 ]), 
        chromatic_adaptation_transform='CAT02', 
        apply_cctf_encoding = apply_cctf_encoding)
    if clip:
        RGB[RGB<0] = 0
        RGB[RGB>1] = 1
    return RGB
     
# -----------------------------------------------------------------------------
# --- Class Processing -------------------------------------------------------
# -----------------------------------------------------------------------------
class Processing(object):
    """
    class Processing: abstract class for processing object

    Methods:
        compute
    """

    def compute(self,image,**kwargs):
        """
        TODO - Documentation de la méthode compute

        Args:
            image (hdrCore.image.Image, Required): input image
                
            kwargs (dict, Optionnal): parameters of processing
                
        Returns:
            (hdrCore.image.Image)

        """
        return copy.deepcopy(image)
# -----------------------------------------------------------------------------
# --- Class tmo_cctf ---------------------------------------------------------
# -----------------------------------------------------------------------------
class tmo_cctf(Processing):
    """
    TODO - Documentation de la classe tmo_cctf
    """

    def compute(self,img,**kwargs):
        """
        CCTF tone mapping operator

        Args:
            img: hdrCore.image.Image
                Required  : hdr image
            kwargs: dict
                Optionnal : parameters
                'function'          : 'sRGB'        str
                
        Returns:
            TODO
                TODO
        """
        function = 'sRGB'
        if 'function' in kwargs: function = kwargs['function']
 
        res = copy.deepcopy(img)

        # can tone map HDR only 
        if (img.type == image.imageType.HDR):

            # encode
            imgRGBprime = colour.cctf_encoding(res.colorData,function=function)

            # update attributes
            res.colorData       = imgRGBprime
            res.type            = image.imageType.SDR
            res.linear          = False
            res.scalingFactor   = 1.0
            res.colorSpace      = colour.models.RGB_COLOURSPACES[function].copy()

        return res
# -----------------------------------------------------------------------------
# --- Class exposure ---------------------------------------------------------
# -----------------------------------------------------------------------------
class exposure(Processing):
    """
    TODO - Documentation de la classe exposure
    """
    
    def compute(self,img,**kwargs):
        """exposure operator.

        Args:
            img (hdrCore.image.Image, Required)  : input image
            kwargs( dict, Optionnal) : parameters
                'EV': float
                
                default value: {'EV': 0.0}
                
        Returns:
            (hdrCore.image.Image): output image
        """
        start= timer()
        defaultEV = 0.0
        if not kwargs: kwargs = {'EV': defaultEV}  # default value
        

        if 'EV' in kwargs : EV = kwargs['EV']
        else:               EV = defaultEV
 
        res = copy.deepcopy(img)

        if EV != defaultEV:
            # exposure is done in linear RGB
            if not res.linear:
                
                if pref.computation == 'python':
                    start = timer()
                    res.colorData =     colour.cctf_decoding(res.colorData, function='sRGB')
                    res.linear =        True

                elif pref.computation == 'numba':
                    start = timer()
                    res.colorData =     numbafun.numba_cctf_sRGB_decoding(res.colorData) # encode to prime
                    res.linear =        True

                elif pref.computation == 'cuda':
                    start = timer()
                    res.colorData =     numbafun.cuda_cctf_sRGB_decoding(res.colorData) # encode to prime
                    res.linear =        True

                dt = timer() - start

            res.colorData =     res.colorData*math.pow(2,EV)

        end = timer()
        if pref.verbose: print (" [PROCESS-PROFILING](",end - start,") >> exposure(",img.name,"):", kwargs)

        return res

    def auto(self,img):
        """
        TODO - Documentation de la méthode auto

        Args:
            img: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """
        minEV, maxEV, step =  -10,10,0.25
        evs = np.linspace(minEV,maxEV,num=int((maxEV-minEV)/step)+1)

        rgb = copy.deepcopy(img.colorData)
        nbPix = rgb.shape[0]*rgb.shape[1]

        if not img.linear:  rgbLinear = colour.cctf_decoding(rgb,function='sRGB')
        else:               rgbLinear = rgb

        bins = np.linspace(0,1,25+1)

        # local method for pool (not static!)
        def evEval(ev):
            """
            TODO - Documentation de la méthode evEval

            Args:
                ev: TODO
                    TODO
                    
            Returns:
                TODO
                    TODO
            """
            rgb_ev = rgbLinear*math.pow(2,ev)
            rgb_ev_prime = colour.cctf_encoding(rgb_ev,function='sRGB')
            rgb_ev_prime[rgb_ev_prime>1] = 1
            XYZ = colour.sRGB_to_XYZ(rgb_ev_prime, apply_cctf_decoding=False)
            Y = utils.ndarray2vector(XYZ)[:,1]
            nphist_ev_prime, npedges = np.histogram(Y, bins)
            nphist_ev_prime = nphist_ev_prime/nbPix
            sumH = np.cumsum(nphist_ev_prime[1:-1])[-1]

            return sumH

        _pool = pathos.multiprocessing.ProcessPool()
        results = _pool.map( evEval, evs)
        sumsH  = list(results)
        
        bestEV = evs[np.argmax(sumsH)]
        if pref.verbose: print('  [PROCESS] >> exposure.auto(',img.name,'):BEST EV:',bestEV)
      
        return {'EV':bestEV}
# -----------------------------------------------------------------------------
# --- Class contrast ---------------------------------------------------------
# -----------------------------------------------------------------------------
class contrast(Processing):
    """
    TODO - Documentation de la classe contrast
    """
    
    def compute(self,img,**kwargs):
        """contrast operator.

        Args:
            img (hdrCore.image.Image,  Required): input image
            kwargs (dict, Optionnal) : parameters
                "contrast": float

                default value: { "contrast": 0 }
                
        Returns:
            (hdrCore.image.Image,  Required): output image
        """
        start = timer()
        defaultContrast =   0.0
        maxContrastFactor = 2.0     ###### 5.0
        if not kwargs: kwargs = { "contrast": defaultContrast }  # default value 


        if 'contrast' in kwargs :   contrastValue = kwargs['contrast']
        else:                       contrastValue = defaultContrast

        res = copy.deepcopy(img)

        if contrastValue != defaultContrast:
            # contrast scaling is computed in prime colorspace
            if img.linear: 
                if pref.computation == 'python':
                    start = timer()
                    res.colorData =     colour.cctf_encoding(res.colorData, function='sRGB') # encode to prime
                    res.linear =        False

                elif pref.computation == 'numba':
                    start = timer()
                    res.colorData =     numbafun.numba_cctf_sRGB_encoding(res.colorData) # encode to prime
                    res.linear =        False

                elif pref.computation == 'cuda':
                    start = timer()
                    res.colorData =     numbafun.cuda_cctf_sRGB_encoding(res.colorData) # encode to prime
                    res.linear =        False

                dt = timer() - start

            # scaling contrast
            contrastValue = contrastValue/100 
            if contrastValue>=0.0:
                scalingFactor = 1*(1-contrastValue)+maxContrastFactor*contrastValue
            else:
                contrastValue = -contrastValue
                scalingFactor = 1*(1-contrastValue)+maxContrastFactor*contrastValue
                scalingFactor = 1/scalingFactor

            res.colorData = scalingFactor*(res.colorData-0.5)+0.5
        
        end=timer()    
        if pref.verbose: print(" [PROCESS-PROFILING] (",end-start,")>> contrast(",img.name,"):", kwargs)

        return res
# -----------------------------------------------------------------------------
# --- Class clip -------------------------------------------------------------
# -----------------------------------------------------------------------------
class clip(Processing):
    """
    TODO - Documentation de la classe clip
    """

    def compute(self,img,**kwargs):
        """clip image color data

        Args:
            img (hdrCore.image.Image, Required): input image image
            kwargs(dict, Optionnal) : parameters
                'min', 'max' : float, float
                
        Returns:
            (hdrCore.image.Image): output image
        """ 

        min, max = 0.0, 1.0
        if 'min' in kwargs: min = kwargs['min']
        if 'max' in kwargs: max = kwargs['max']
        
        res = copy.deepcopy(img)
        res.colorData[res.colorData>max] = max
        res.colorData[res.colorData<min] = min

        return res
# -----------------------------------------------------------------------------
# --- Class ColorSpaceTransform ----------------------------------------------
# -----------------------------------------------------------------------------
class ColorSpaceTransform(Processing):
    """
    TODO - Documentation de la colorspace transorm processing
    """

    def compute(self,img,**kwargs):
        """
        Color space transform operator

        Args:
            img (hdrCore.image.Image, Required)  : input image
            kwargs (dict,Optionnal) : parameters
                TODO
                
        Returns:
            TODO
                TODO
        """ 
        # first create a copy
        res = copy.deepcopy(img)
        if not kwargs: print("WARNING[Processing.ColorSpaceTransform(",img.name,"):", "no destination colour space >> return a copy of image]")
        else:
            if not 'dest'in kwargs: print("WARNING[Processing.ColorSpaceTransform(",img.name,"):", "no 'dest' colour space >> return a copy of image]")
            else: # -> 'colorSpace' found
                if  kwargs['dest'] == 'Lab': # DEST: Lab
                    currentCS = img.colorSpace.name
                    # sRGB to Lab
                    if currentCS=="sRGB": # sRGB -> Lab
                        apply_cctf_decoding=True if not img.linear else False
                        
                        RGB = res.colorData
                        XYZ = colour.sRGB_to_XYZ(RGB, illuminant=np.array([ 0.3127, 0.329 ]), chromatic_adaptation_transform='CAT02', apply_cctf_decoding=apply_cctf_decoding)           
                        Lab = colour.XYZ_to_Lab(XYZ, illuminant=np.array([ 0.3127, 0.329 ]))
                        res.colorData, res.linear, res.colorSpace  = Lab, None,image.ColorSpace.Lab()

                    elif currentCS=="XYZ": # XYZ -> Lab 
                        XYZ = res.colorData
                        Lab = colour.XYZ_to_Lab(XYZ, illuminant=np.array([ 0.3127, 0.329 ]))
                        res.colorData, res.linear,  = Lab, None, image.ColorSpace.XYZ()
                              
                    elif currentCS == "Lab": # Lab -> Lab                       
                        pass # return a copy

                elif kwargs['dest'] == 'sRGB': # DEST: sRGB
                    currentCS = img.colorSpace.name
                    if currentCS=="Lab":  # Lab -> sRGB                                                               
                        Lab = res.colorData
                        XYZ = colour.Lab_to_XYZ(Lab, illuminant=np.array([ 0.3127, 0.329 ]))
                        apply_cctf_encoding = False if img.type == image.imageType.HDR  else  True
                        sRGB = colour.colour.XYZ_to_sRGB(XYZ, illuminant=np.array([ 0.3127, 0.329 ]), chromatic_adaptation_transform='CAT02', apply_cctf_encoding=apply_cctf_encoding)
                        res.colorData, res.colorSpace, res.linear = sRGB, image.ColorSpace.sRGB(), not apply_cctf_encoding
                    
                    elif currentCS == "XYZ":# XYZ -> sRGB 
                        XYZ = res.colorData
                        apply_cctf_encoding = False if (img.type == image.imageType.HDR)  else True
                        sRGB = colour.colour.XYZ_to_sRGB(XYZ, illuminant=np.array([ 0.3127, 0.329 ]), chromatic_adaptation_transform='CAT02', apply_cctf_encoding=apply_cctf_encoding)
                        res.colorData, res.colorSpace, res.linear = sRGB, image.ColorSpace.sRGB(), not apply_cctf_encoding
    
                    elif currentCS == "sRGB": # sRGB -> sRGB
                        pass # return a copy             
                    else:
                        print("WARNING[Processing.ColorSpaceTransform(",img.name,"):", "'dest' colour space:",kwargs['dest'] , "not yet implemented !]")

                elif kwargs['dest'] == 'XYZ': # DEST: XYZ
                    currentCS = img.colorSpace.name
                    if currentCS=="sRGB": # sRGB to XYZ                                                         
                        apply_cctf_decoding=True if  (img.type == image.imageType.SDR) and (not img.linear) else False
                        RGB = res.colorData
                        XYZ = colour.sRGB_to_XYZ(RGB, illuminant=np.array([ 0.3127, 0.329 ]), chromatic_adaptation_transform='CAT02', apply_cctf_decoding=apply_cctf_decoding)           
                        res.colorData,res.linear , res.colorSpace = XYZ, True, image.ColorSpace.XYZ()
                        
                    elif currentCS=="XYZ": # XYZ to XYZ                                                         
                         pass # return a copy
                    
                    elif currentCS == "Lab": # Lab to XYZ
                        Lab = res.colorData
                        XYZ = colour.Lab_to_XYZ(Lab, illuminant=np.array([ 0.3127, 0.329 ]))
                        res.colorData, res.linear, res.colorSpace = XYZ, True, image.ColorSpace.buildXYZ()

        return res
# -----------------------------------------------------------------------------
# --- Class resize -----------------------------------------------------------
# -----------------------------------------------------------------------------
class resize(Processing):
    """
    TODO - Documentation de la classe Resize image processing
    """
    
    def compute(self,img, size=(None,None),anti_aliasing=False):
        """
        Resize operator.

        Args:
            img: hdrCore.image.Image
                Required : hdr image
            size: TODO
                TODO
            anti_aliasing: boolean
                TODO
                
        Returns:
            TODO
                TODO
        """
        res = copy.deepcopy(img)
        y, x, c =  tuple(res.colorData.shape)
        ny,nx = size
        if nx and (not ny): 
            factor = nx/x
            res.colorData = skimage.transform.resize(res.colorData, (int(y * factor),nx ), anti_aliasing)
            res.shape = res.colorData.shape
        elif (not nx) and ny:
            factor = ny/y
            res.colorData = skimage.transform.resize(res.colorData, (ny,int(x * factor)), anti_aliasing)
            res.shape = res.colorData.shape
        elif nx and ny:
            res.colorData = skimage.transform.resize(res.colorData, (ny,nx), anti_aliasing)
            res.shape = res.colorData.shape
        elif (not nx) and (not ny):
            ny=400
            factor = ny/y
            res.colorData = skimage.transform.resize(res.colorData, (ny,int(x * factor)), anti_aliasing)
            res.shape = res.colorData.shape
        return res
# -----------------------------------------------------------------------------
# --- Class Ycurve -----------------------------------------------------------
# -----------------------------------------------------------------------------
class Ycurve(Processing):
    """
    TODO - Documentation de la classe Ycurve
    """
    
    def compute(self,img,**kwargs):
        """
        compute: compute Ycurve according to parameters.

        Args:
            img (hdrCore.image.Image, Required): image
            kwargs (dict,Optionnal): parameters gieven as dict
                
        Returns:
            (hdrCore.image.Image, Required): image
                result of Ycurve processing
        """ 
        start =  timer()
        defaultControlPoints = {'start':[0,0], 
                                'shadows': [10,10], 
                                'blacks': [30,30], 
                                'mediums': [50,50], 
                                'whites': [70,70], 
                                'highlights': [90,90], 
                                'end': [100,100]}

        if not kwargs: kwargs = defaultControlPoints  # default value 


        # results image
        res = copy.deepcopy(img)

        if kwargs != defaultControlPoints:

            if img.linear: 
                if pref.computation == 'python':
                    start = timer()
                    res.colorData =     colour.cctf_encoding(res.colorData, function='sRGB') # encode to prime
                    res.linear =        False

                elif pref.computation == 'numba':
                    start = timer()
                    res.colorData =     numbafun.numba_cctf_sRGB_encoding(res.colorData) # encode to prime
                    res.linear =        False

                elif pref.computation == 'cuda':
                    start = timer()
                    res.colorData =     numbafun.cuda_cctf_sRGB_encoding(res.colorData) # encode to prime
                    res.linear =        False

                dt = timer() - start

            colorDataY =    sRGB_to_XYZ(res.colorData, apply_cctf_decoding=False)[:,:,1] 
            # change for multi-threading computation
            # Ymax =          np.amax(colorDataY)*100
            # extendedEnd =   [Ymax, kwargs['end'][1]]
            extendedEnd =   [200, kwargs['end'][1]]

            # create curve adn get y-curve
            curve =         BSpline.Curve()
            curve.degree =  2
            points =        None
            curve.ctrlpts =     copy.deepcopy([kwargs['start'], kwargs['shadows'], kwargs['blacks'],  kwargs['mediums'], kwargs['whites'], kwargs['highlights'], extendedEnd])
            curve.knotvector =  utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
            # evaluate curve and get points
            points = np.asarray(curve.evalpts)/100
            Y = points[:,0]
            FY = points[:,1]

            # check validity: same shape
            if Y.shape == FY.shape:
                colorDataFY = np.interp(colorDataY, Y,FY)

                # remove zeros
                Ymin = np.amin(colorDataY[colorDataY>0])
                colorDataY[colorDataY==0] = Ymin

                # transform colorData
                res.colorData[:,:,0] = res.colorData[:,:,0]*colorDataFY/colorDataY
                res.colorData[:,:,1] = res.colorData[:,:,1]*colorDataFY/colorDataY
                res.colorData[:,:,2] = res.colorData[:,:,2]*colorDataFY/colorDataY
        
        end = timer()        
        if pref.verbose: print(" [PROCESS-PROFILING] (",end - start,")>> Ycurve(",img.name,"):", kwargs)

        return res
# -----------------------------------------------------------------------------
# --- Class saturation -------------------------------------------------------
# -----------------------------------------------------------------------------
class saturation(Processing):
    """
    TODO - Documentation de la classe saturation
    """
    
    def compute(self,img,**kwargs):
        """saturation operator

        Args:
            img (hdrCore.image.Image,Required):  input image
            kwargs (dict, Optionnal) : parameters
                'saturation': float
                'method': str
                    'method' parameter must be gamma 
                default value: {'saturation': 0.0, 'method': 'gamma'}
                
        Returns:
            (hdrCore.image.Image): output image
        """ 
        start=timer()
        defaultValue= {'saturation': 0.0, 'method': 'gamma'}

        if not kwargs: kwargs = defaultValue  # default value 


        # results image
        res = copy.deepcopy(img)

        value = kwargs["saturation"]
        if value != defaultValue['saturation']:

            # go to Lab then Lch
            if img.linear: 
                colorLab = sRGB_to_Lab(res.colorData, apply_cctf_decoding=False)
            else:
                colorLab = sRGB_to_Lab(res.colorData, apply_cctf_decoding=True)
            colorLCH = colour.Lab_to_LCHab(colorLab)

            # saturation in Lch (chroma as saturation)
            gamma = 1/((value/25)+1) if value >= 0 else (-value/25)+1
            colorLCH[:,:,1] =np.power(colorLCH[:,:,1]/100, gamma)*100


            #colorRGB_sat = Lab_to_sRGB(colour.LCHab_to_Lab(colorLCH), apply_cctf_encoding=True)
            #res.colorData = colorRGB_sat
            #res.linear = False
            res.colorData = colorLCH
            res.linear = False
            res.colorSpace = image.ColorSpace.build('Lch')


        end = timer()
        if pref.verbose: print(" [PROCESS-PROFILING] (",end - start,")>> saturation(",img.name,"):", kwargs)

        return res
# -----------------------------------------------------------------------------
# --- Class colorEditor ------------------------------------------------------
# -----------------------------------------------------------------------------
class colorEditor(Processing):
    """
    TODO - Documentation de la classe colorEditor
    """
    
    def compute(self,img, **kwargs):
        """color editor operator

        Args:
            img (hdrCore.image.Image, Required): input image
            kwargs (dict, Optionnal): parameters
                default value= { 'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)}, 
                                'tolerance': 0.1,
                                'edit': {'hue':0.0,'exposure':0.0,'contrast':0.0,'saturation':0.0}, 
                                'mask': False}                
                
        Returns:
            (hdrCore.image.Image): output image
                
        """
        start = timer()
        defaultValue= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)}, 
                       'tolerance': 0.1,
                       'edit': {'hue':0.0,'exposure':0.0,'contrast':0.0,'saturation':0.0}, 
                       'mask': False}

        if not kwargs: kwargs = defaultValue  # default value
        if not ('selection' in kwargs): kwargs['selection'] =   defaultValue['selection']
        if not ('tolerance' in kwargs): kwargs['tolerance'] =   defaultValue['tolerance']
        if not ('edit' in kwargs):      kwargs['edit'] =        defaultValue['edit']
        if not ('mask' in kwargs):      kwargs['mask'] =         defaultValue['mask']


        # results image
        res = copy.deepcopy(img)

        # computing
        if kwargs != defaultValue:
            colorRGB = None
            if res.colorSpace.name == 'Lch':
                colorLCH = res.colorData
            elif res.colorSpace.name == 'sRGB':

                covnStart = timer()
                if res.linear: 

                    colorLab = sRGB_to_Lab(res.colorData, apply_cctf_decoding=False)
                    colorLCH = colour.Lab_to_LCHab(colorLab)

                else:

                    colorLab = sRGB_to_Lab(res.colorData, apply_cctf_decoding=True)
                    colorLCH = colour.Lab_to_LCHab(colorLab)
                covnEnd = timer()

            # selection from colorLCH
            colorDataHue =          copy.deepcopy(colorLCH[:,:,2])
            colorDataChroma =       copy.deepcopy(colorLCH[:,:,1])
            colorDataLightness =    copy.deepcopy(colorLCH[:,:,0])

            # selection mask
            hMin, hMax = kwargs['selection']['hue'] if 'hue' in kwargs['selection'].keys() else defaultValue['selection']['hue']
            cMin, cMax = kwargs['selection']['chroma'] if 'chroma' in kwargs['selection'].keys() else defaultValue['selection']['chroma']
            lMin, lMax = kwargs['selection']['lightness']if 'hue' in kwargs['selection'].keys() else defaultValue['selection']['lightness']
            # take into account Chroma, Lightness range
            cMax = cMax*max(100.0,np.amax(colorDataChroma))/100.0
            lMax = lMax*max(100.0,np.amax(colorDataLightness))/100.0

            # tolerance
            hueTolerance = kwargs['tolerance']*360      # hue range ~ 360
            chromaTolerance = kwargs['tolerance']*100   # chroma range ~ 100
            lightTolerance = kwargs['tolerance']*100    # lightness range ~ 100


            maskStart = timer()

            lightnessMask =     utils.NPlinearWeightMask(colorDataLightness, lMin, lMax, lightTolerance)
            chromaMask =        utils.NPlinearWeightMask(colorDataChroma, cMin, cMax, chromaTolerance)
            hueMask =           utils.NPlinearWeightMask(colorDataHue, hMin, hMax, hueTolerance)

            maskEnd = timer()

            mask = np.minimum(lightnessMask, np.minimum(chromaMask,hueMask))
            compMask = 1.0 - mask
            # hueShift (in Lch)
            hueShift =  kwargs['edit']['hue']  if 'hue' in kwargs['edit'].keys() else defaultValue['edit']['hue']
            if hueShift != 0.0:
                colorLCH[:,:,2] = ((colorLCH[:,:,2]+hueShift)%360)*mask + colorLCH[:,:,2]*compMask

            # saturation (in Lch)
            saturation = kwargs['edit']['saturation'] if 'saturation' in kwargs['edit'].keys() else defaultValue['edit']['saturation']
            if saturation != 0 :
                gamma = 1/((saturation/25)+1) if saturation >= 0 else (-saturation/25)+1
                colorLCH[:,:,1] =np.power(colorLCH[:,:,1]/100, gamma)*100*mask + colorLCH[:,:,1]*compMask

            # exposure (in RGB)
            ev =  kwargs['edit']['exposure'] if 'exposure' in kwargs['edit'].keys() else defaultValue['edit']['exposure']
            if ev != 0.0 :
                colorRGB = Lch_to_sRGB(colorLCH,apply_cctf_encoding=False, clip=False)
                colorRGBev = copy.deepcopy(colorRGB*math.pow(2,ev))
                colorRGBev[:,:,0] = colorRGBev[:,:,0]*mask
                colorRGBev[:,:,1] = colorRGBev[:,:,1]*mask
                colorRGBev[:,:,2] = colorRGBev[:,:,2]*mask

                colorRGB[:,:,0] = colorRGB[:,:,0]*compMask
                colorRGB[:,:,1] = colorRGB[:,:,1]*compMask
                colorRGB[:,:,2] = colorRGB[:,:,2]*compMask
                colorRGB = colorRGB + colorRGBev

            # contrast (in RGB prime)
            con =  kwargs['edit']['contrast'] if 'exposure' in kwargs['edit'].keys() else defaultValue['edit']['contrast']
            if con != 0 :
                con = con/100
                maxContrastFactor = 2.0
                if con>=0.0:
                    scalingFactor = 1*(1-con)+maxContrastFactor*con
                else:
                    con = -con
                    scalingFactor = 1*(1-con)+maxContrastFactor*con
                    scalingFactor = 1/scalingFactor

                pivot = math.pow(2,ev)*(lMin+lMax)/2/100

                if not isinstance(colorRGB, np.ndarray):    colorRGB = Lch_to_sRGB(colorLCH,apply_cctf_encoding=True, clip=False)
                else :                                      colorRGB = colour.cctf_encoding(colorRGB, function='sRGB')
                
                colorRGBcon = (colorRGB-pivot)*scalingFactor+pivot

                colorRGB[:,:,0] = colorRGBcon[:,:,0]*mask+ colorRGB[:,:,0]*compMask
                colorRGB[:,:,1] = colorRGBcon[:,:,1]*mask+ colorRGB[:,:,1]*compMask
                colorRGB[:,:,2] = colorRGBcon[:,:,2]*mask+ colorRGB[:,:,2]*compMask

                colorRGB = colour.cctf_decoding(colorRGB, function='sRGB')

            # final step
            if not isinstance(colorRGB, np.ndarray): colorRGB = Lch_to_sRGB(colorLCH,apply_cctf_encoding=False, clip=False)
            res.colorData = colorRGB
            res.colorSpace = image.ColorSpace.build('sRGB')
            res.linear = True

        else:
            if res.colorSpace.name == 'Lch':
                colorLCH = res.colorData
                # return to RGB (linear)
                colorRGB = Lch_to_sRGB(colorLCH,apply_cctf_encoding=False, clip=False)
                res.colorData = colorRGB
                res.colorSpace = image.ColorSpace.build('sRGB')
                res.linear = True

        showMask = kwargs['mask']
        if showMask:
            res.colorData[:,:,0] = copy.deepcopy(mask)
            res.colorData[:,:,1] = copy.deepcopy(mask)
            res.colorData[:,:,2] = copy.deepcopy(mask)

            res.colorSpace = image.ColorSpace.build('sRGB')
            res.linear = False

        end = timer()
        if pref.verbose: print(" [PROCESS-PROFILING](",end - start,") >> colorEditor(",img.name,"):", kwargs)

        return res
# -----------------------------------------------------------------------------
# --- Class lightnessMask ----------------------------------------------------
# -----------------------------------------------------------------------------
class lightnessMask(Processing):
    """
    TODO - Documentation de la classe lightnessMask
    """
    
    def compute(self, img, **kwargs):
        """
        Lightness Mask operator.

        Args:
            img: hdrCore.image.Image
                Required  : hdr image
            kwargs: dict
                Optionnal : parameters
                TODO
                
        Returns:
            TODO
                TODO
        """
        start = timer()
        defaultMask = { 'shadows': False, 'blacks': False, 'mediums': False, 'whites': False, 'highlights': False}
        rangeMask = {   'shadows': [0,20], 
                         'blacks': [20,40], 
                         'mediums': [40,60], 
                         'whites': [60,80], 
                         'highlights': [80,100]}
        maskColor = {'shadows': [0,0,1], 
                     'blacks': [0,1,1], 
                     'mediums': [0,1,0], 
                     'whites': [1,1,0], 
                     'highlights': [1,0,0]}
        if not kwargs: kwargs = defaultMask  # default value 

        # results image
        res = copy.deepcopy(img)

        if kwargs != defaultMask:

            if img.linear: 
                res.colorData = colour.cctf_encoding(res.colorData, function='sRGB') # encode to prime   
                res.linear = False

            colorDataY = sRGB_to_XYZ(res.colorData, apply_cctf_decoding=False)[:,:,1]
            mask = copy.deepcopy(res.colorData)

            for key in rangeMask.keys():
                if kwargs[key]: # mask on
                    mask[(colorDataY >= rangeMask[key][0]/100)*(colorDataY<rangeMask[key][1]/100),:] = np.asarray(maskColor[key])

            res.colorData = mask
        
        end = timer()
        if pref.verbose: print(" [PROCESS-PROFILING](",end - start,") >> lightnessMask(",res.name,"):", kwargs)

        return res
# -----------------------------------------------------------------------------
# --- Class geometry ---------------------------------------------------------
# -----------------------------------------------------------------------------
class geometry(Processing):
    """
    TODO - Documentation de la classe geometry
    """
    
    def compute(self, img, **kwargs): 
        """geometry operator.

        Args:
            img (hdrCore.image.Image,Required): input image
            kwargs (dict,Optionnal) : parameters
                'ratio': (int,int)
                'up': int
                'rotation': float

                default value = { 'ratio': (16,9), 'up': 0,'rotation': 0.0}
                
        Returns:
            (hdrCore.image.Image,Required): input image
        """ 
        start = timer()
        defaultValue = { 'ratio': (16,9), 'up': 0,'rotation': 0.0}
        if not kwargs: kwargs = defaultValue  # default value 
        ratio =     kwargs['ratio']     if 'ratio' in kwargs.keys()     else defaultValue['ratio']
        up =        kwargs['up']        if 'up' in kwargs.keys()        else defaultValue['up']
        rotation =  kwargs['rotation']  if 'rotation' in kwargs.keys()  else defaultValue['rotation']

        # results image
        res = copy.deepcopy(img)

        ##if kwargs != defaultValue:
        h,w, c = res.colorData.shape
        imgRatio = w/h

        if int(imgRatio*1000) != int(ratio[0]/ratio[1]*1000):
            if imgRatio < (ratio[0]/ratio[1]):
                hh16x9 = int(w*ratio[1]/ratio[0]/2)
                ch = h//2
                up = int((h//2-hh16x9)*up/100)
                res.colorData = res.colorData[(ch-hh16x9-up):(ch+hh16x9-up),:,:]
                res.shape = res.colorData.shape
            else:
                ww16x9 = int(h*ratio[0]/ratio[1]/2)
                ch = w//2
                res.colorData = res.colorData[:,(ch-ww16x9):(ch+ww16x9),:]
                res.shape = res.colorData.shape

        if rotation != 0 :
            res.colorData = skimage.transform.rotate(res.colorData, rotation, clip = False, resize=False)
            h,w, _ = res.colorData.shape
            hh,ww = utils.croppRotated(h,w,rotation)
            res.colorData = res.colorData[int(h/2-hh/2):int(h/2+hh/2), int(w/2-ww/2):int(w/2+ww/2),:]
            res.shape = res.colorData.shape

        end = timer()
        if pref.verbose: print(" [PROCESS-PROFILING] (",end-start,")>> geometry(",res.name,"):", kwargs)

        return res
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Class ProcessPipe ------------------------------------------------------
# -----------------------------------------------------------------------------
class ProcessPipe(object):
    """
    class ProcessPipe: pipeline of process nodes
        defines the pipeline of image procesing objects. 
    
    Attributes:
        originalImage (hdrCore.image.Image):
        __inputImage (hdrCore.image.Image):
        __outputImage (hdrCore.image.Image):
        processNodes ([ProcessNode]):
        previewHDR (bool):
        previewHDR_process ():

    Class Attributes:
        autoResize (boolean): True resize automatically image for faster computation
        maxSize (int): 
        maxWorking (int):       

    Methods:
        append:                 (int) append a process node (ProcessNode) to process pipe (self)
        getName:                (str) return image name associated to processpipe
        setImage:               ()
        getInputImage           ()
        compute                 ()
        setParameters           ()
        getParameters           ()
        getProcessNodeByName    ()
        __repr__                (str)
        __str__                 (str)
        updateProcessPipeMetadata ()
        updateHDRuseCase        ()
        export                  ()
    """
    
    # autoresizing for fast computation
    autoResize =    True
    maxSize =       1200 
    maxWorking =    1200 #800
     
    # -------------------------------------------------------------------------
    # --- Class ProcessNode --------------------------------------------------
    # -------------------------------------------------------------------------
    class ProcessNode(object):
        """encapsulates a Processing object to create processpipe

        Attributes:
            name (str): name of ProcessNode
            process (hdrCore.processing.Processing):
            params(dict):
            defaultParams (dict):
            requireUpdate (bool):
            outputImage (hdrCore.image.Image):   
            
        Methods:
            compute 
            condCompute
            setParameters
            getParameters
            toDict
        """

        
        id=0
        
        def __init__(self,process,paramDict=None,name=None):

            self.name = 'noname'
            if not name:
                self.name = 'process_'+str(ProcessNode.id)
                ProcessNode.id += 1
            else:
                self.name = name
            self.process = process
            self.params = paramDict
            self.defaultParams = copy.deepcopy(paramDict)
            self.requireUpdate = True # require a first process
            self.outputImage = None # store results image (Image)

        def compute(self,img):

            self.outputImage = self.process.compute(img,**self.params)
            self.requireUpdate = False

        def condCompute(self,img):

           if self.requireUpdate: self.compute(img)
           pass

        def setParameters(self,paramDict):

            if pref.verbose: print(" [PROCESS] >> ProcessNode.setParameters(",self.name,"):",paramDict)
            self.params=paramDict
            self.requireUpdate = True

        def getParameters(self):

            return self.params

        def toDict(self):

            return {self.name: self.params}
    # -------------------------------------------------------------------------
    # --- End of ProcessNode -------------------------------------------
    # -------------------------------------------------------------------------
    
    def __init__(self):
        """
        TODO - Documentation de la méthode __init__
        
        /!\ - Les constructeurs n'apparaissent pas dans la doc générées par sphinx.
        """
        self.originalImage = None # 
        self.__inputImage = None
        self.__outputImage = None
        self.processNodes = []

        self.previewHDR = True
        self.previewHDR_process = None

    def append(self,process,paramDict=None,name=None):
        """
        TODO - Documentation de la méthode append

        Args:
            process: TODO
                TODO
            paramDict: TODO
                TODO
            name: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """
        if isinstance(process,Processing): process = ProcessPipe.ProcessNode(process,paramDict,name) # encapsulate process in processNode
        self.processNodes.append(process)
        return len(self.processNodes)-1 # return index of process (list[index])

    def getName(self):
        """return name of input image.
        
        Returns:
            (str)
        """
        return self.__outputImage.name

    def setImage(self,img):
        """set the input image to the process-pipeline:
            (1) a copy of the image is set to 'originalImage'
            (2) if ProcessPipe.autoResize: the image is resized
            (3) the (resize) is set to '__inputImage'
            (4) a copy of the resized image is set to '__outputImage' (for display)
            (5) initialize processpipe using 'img.metadata'  
            (6) for all processes in the pipe 'requireUpdate' is set to True

        Args:
            img (hdrCore.image.Image, Required) : input image

        Returns:
            
        """
        if pref.verbose: print(" [PROCESS] >> ProcessPipe.setImage(",img.name,")")

        # resize input for faster computation
        if ProcessPipe.autoResize:
            height, width, channels = img.shape
            if (height>= width) and (height>ProcessPipe.maxWorking):
                img = img.process(resize(),size=(ProcessPipe.maxWorking,None))

            elif (width>=height) and (width>ProcessPipe.maxWorking):   
                img = img.process(resize(),size=(None,ProcessPipe.maxWorking))

        self.originalImage= copy.deepcopy(img)

        # a copy is set as __outputImage
        self.__outputImage = copy.deepcopy(img)
     
        if not img.linear: 


            if pref.computation == 'python':
                start = timer()
                img.colorData =     np.float32(colour.cctf_decoding(img.colorData, function='sRGB'))
                img.linear =        True

            elif pref.computation == 'numba':
                start = timer()
                img.colorData= numbafun.numba_cctf_sRGB_decoding(img.colorData)
                img.linear =        True

            elif pref.computation == 'cuda':
                start = timer()
                img.colorData =     numbafun.cuda_cctf_sRGB_decoding(img.colorData) # encode to prime
                img.linear =        True

            dt = timer() - start

        # input image is set as __inputImage
        self.__inputImage = img

        # requireUpdate is set to True
        for processNode in self.processNodes: processNode.requireUpdate = True

        # recover medata to initialize processPipe
        if 'processpipe' in img.metadata.metadata:
            processpipeMetadata = img.metadata.metadata['processpipe']

            if isinstance(processpipeMetadata,list):
                for pMeta in processpipeMetadata:

                    key = list(pMeta.keys())[0]
                    param = pMeta[key]
                    idProcess = self.getProcessNodeByName(key)
                    if idProcess != -1:
                        self.setParameters(idProcess,param)

    def setOutput(self, img):
        """setOuput: set the output image
        """
        self.__outputImage = copy.deepcopy(img)
        pass

    def getInputImage(self):
        """return input image
        
        Returns:
            (hdrCore.image.Image)
        """
        return self.__inputImage

    def getImage(self,toneMap=True):
        """
        TODO - Documentation de la méthode getImage

        Args:
            toneMap: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """
        if isinstance(self.originalImage, image.Image): # if pipe has an image
            # conditionnal encoding or decoding to prime, linear

            if (not self.originalImage.linear) and self.__outputImage.linear:
                self.__outputImage.colorData = colour.cctf_encoding(self.__outputImage.colorData, function='sRGB')
                self.__outputImage.linear =  False

                if pref.verbose: print(" [PROCESS] >> ProcessPipe.getImage(",self.__outputImage.name,", toneMap:",toneMap,"): encode to sRGB !")

            elif self.__outputImage.isHDR() and self.__outputImage.linear and toneMap:
                self.__outputImage.colorData = colour.cctf_encoding(self.__outputImage.colorData, function='sRGB')
                self.__outputImage.linear =  False

                if pref.verbose: print(" [PROCESS] >> ProcessPipe.getImage(",self.__outputImage.name,", ,toneMap:",toneMap,"): tone map using cctf encoding !")

            elif self.__outputImage.isHDR() and (not self.__outputImage.linear) and (not toneMap):
                self.__outputImage.colorData = colour.cctf_decoding(self.__outputImage.colorData, function='sRGB')
                self.__outputImage.linear =  True

                if pref.verbose: print(" [PROCESS] >> ProcessPipe.getImage(",self.__outputImage.name,", toneMap:",toneMap,"): decoding to linear colorspace !")

            elif (not self.__outputImage.linear) and (not toneMap):
                self.__outputImage.colorData = colour.cctf_decoding(self.__outputImage.colorData, function='sRGB')
                self.__outputImage.linear =  True

                if pref.verbose: print(" [PROCESS] >> ProcessPipe.getImage(",self.__outputImage.name,", toneMap:",toneMap,"): decoding to linear colorspace !")

            else:
                if pref.verbose: print(" [PROCESS] >> ProcessPipe.getImage(",self.__outputImage.name,", toneMap:",toneMap,"): just return output !")

            return self.__outputImage
        else: return None

    def compute(self,progress=None):
        """compute the processpipe

        Args:
            progress: (object with showMessage and repaint method) object used to display progress

        Returns:
                
        """
        if self.__inputImage:

            if len(self.processNodes)>0: 
                # first node
                if progress:
                    progress.showMessage('computing: '+self.processNodes[0].name+' start!')
                    progress.repaint()
                self.processNodes[0].condCompute(self.__inputImage)
                if progress:
                    progress.showMessage('computing: '+self.processNodes[0].name+' done!')
                    progress.repaint()
                # other nodes
                for i,processNode in enumerate(self.processNodes[1:]):
                    if progress:
                        progress.showMessage('computing: '+processNode.name+' start!')
                        progress.repaint()
                    processNode.condCompute(self.processNodes[i].outputImage)
                    if progress:
                        progress.showMessage('computing: '+processNode.name+' done!')
                        progress.repaint()
            self.__outputImage=self.processNodes[-1].outputImage

    def setParameters(self,id,paramDicts):
        """
        TODO - Documentation de la méthode setParameters

        Args:
            id: TODO
                TODO
            paramDicts: TODO
                TODO
        """
        self.processNodes[id].setParameters(paramDicts)
        for processNode in self.processNodes[id:]: processNode.requireUpdate = True
        self.updateProcessPipeMetadata()

    def getParameters(self,id):
        """
        TODO - Documentation de la méthode getParameters

        Args:
            id: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """
        return self.processNodes[id].getParameters()

    def getProcessNodeByName(self,name):
        """
        TODO - Documentation de la méthode getProcessNodeByName

        Args:
            name: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """
        id = -1
        for i, process in enumerate(self.processNodes):
            if process.name == name:
                id  = i
                break
        return id

    def __repr__(self):
        """
        TODO - Documentation de la méthode __repr__
                
        Returns:
            TODO
                TODO
        """
        res =   "<class ProcessPipe: \n"+ \
                "\t inputImage: "
        if self.__inputImage:
            res += self.__inputImage.name + "\n"
        else:
            res += "None" +"\n"
        for i,p in enumerate(self.processNodes):
            res += "\t["+str(i)+"]:"+p.name+"( params: "+str(p.params)+"| requireUpdate: "+str(p.requireUpdate)+") \n"
        return res+">"

    def __str__(self):
        """
        TODO - Documentation de la méthode __str__
                
        Returns:
            TODO
                TODO
        """
        return self.__repr__()

    def toDict(self):
        """
        TODO - Documentation de la méthode toDict
                
        Returns:
            TODO
                TODO
        """
        res = []
        for p in self.processNodes: res.append(p.toDict())
        return res

    def updateProcessPipeMetadata(self):
        """
        TODO - Documentation de la méthode updateProcessPipeMetadata
        """
        ppMeta = self.toDict()
        if pref.verbose: print(" [PROCESS] >> ProcessPipe.updateMetadata(","):",ppMeta)
        if isinstance(self.originalImage,image.Image):  self.originalImage.metadata.metadata['processpipe'] =   copy.deepcopy(ppMeta)
        if isinstance(self.__inputImage,image.Image):   self.__inputImage.metadata.metadata['processpipe'] =    copy.deepcopy(ppMeta)
        if isinstance(self.__outputImage,image.Image):  self.__outputImage.metadata.metadata['processpipe'] =   copy.deepcopy(ppMeta)

    def updateUserMeta(self,tagRootName,meta):
        """
        TODO - Documentation de la méthode updateUserMeta

        Args:
            hdrmeta: TODO
                TODO
        """
        if pref.verbose: print(" [PROCESS] >> ProcessPipe.updateUserMeta(",")")
        if isinstance(self.originalImage,image.Image):  self.originalImage.metadata.metadata[tagRootName] =   copy.deepcopy(meta)
        if isinstance(self.__inputImage,image.Image):   self.__inputImage.metadata.metadata[tagRootName] =    copy.deepcopy(meta)
        if isinstance(self.__outputImage,image.Image):  self.__outputImage.metadata.metadata[tagRootName] =   copy.deepcopy(meta)

    def export(self,dirName,size=None,to=None,progress=None):
        """
        TODO - Documentation de la méthode export

        Args:
            dirName: TODO
            size: TODO
            to: TODO
            progress: TODO
                
        Returns:
            (hdrCore.image.Image)
        """
        # recover input and processpipe metadata
        input = copy.deepcopy(self.originalImage)
        input.metadata.metadata['processpipe'] = self.toDict()
        input.metadata.save()

        # load full size image
        img = image.Image.read(self.originalImage.path+'/'+self.originalImage.name)
        if size: img = img.process(resize(),size=(None, size[1]))

        ProcessPipe.autoResize = False # set off autoresize

        self.setImage(img)

        self.compute(progress=progress)
        ###### res = hdrCore.coreC.coreCcompute(img, self)

        res = self.getImage(toneMap=False)
        res = res.process(clip())

        res.metadata = copy.deepcopy(img.metadata)                  # exif, hdr use case, ...
        res.metadata.metadata['processpipe'] = None                  # reset process pipe  
        
        ProcessPipe.autoResize = True# restore autoresize
        if to:
            res.colorData = res.colorData*to['scaling']
            res.metadata.metadata['display'] = to['tag']     # set display

        if dirName:
            pathExport = os.path.join(dirName, img.name[:-4]+to['post']+'.hdr')
            res.write(pathExport)

        #restore input
        self.setImage(input)
        self.compute()

        return res
