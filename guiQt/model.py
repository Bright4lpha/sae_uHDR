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
# --- Package hdrGUI ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
package hdrGUI consists of the classes for GUI.

"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------

import os, colour, copy, json, time, sklearn.cluster, math
import pathos.multiprocessing, multiprocessing, functools
import numpy as np
from geomdl import BSpline
from geomdl import utilities

from datetime import datetime

import hdrCore.image, hdrCore.utils, hdrCore.aesthetics, hdrCore.image
from . import controller, thread
import hdrCore.processing, hdrCore.quality
import preferences.preferences as pref

from PyQt5.QtCore import QRunnable

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageWidgetModel(object):
    """ image gui model """

    def __init__(self, controller):

        self.controller = controller
        self.image = None # numpy array or hdrCore.image.Image

    def setImage(self,image): self.image = image

    def getColorData(self): 
        if isinstance(self.image, np.ndarray):
            return self.image
        elif isinstance(self.image, hdrCore.image.Image):
            return self.image.colorData
# ------------------------------------------------------------------------------------------
# --- class ImageGalleryModel --------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageGalleryModel:
    """ ImageGalleryModel:management of image gallery

        Attributes:
            controller (ImageGalleryController): parent of self
            imagesFilenames (list[str]): list of image filenames
            processPipes (list[hdrCore.porocessing.ProcessPipe]): list of process-pipes associated to images
            _selectedImage (int): index of current (selected) process-pipe

            aestheticsModels (list[hdrCore.aesthetics.MultidimensionalImageAestheticsModel])

        Methods:
            setSelectedImage
            selectedImage
            getSelectedProcessPipe
            setImages
            loadPage
            save
            getFilenamesOfCurrentPage
            getProcessPipeById
    """

    def __init__(self, _controller):
        if pref.verbose:  print(" [MODEL] >> ImageGalleryModel.__init__()")

        self.controller = _controller
        self.imageFilenames = []
        self.processPipes = []
        self._selectedImage= -1

        self.aesthetics = []

    def setSelectedImage(self,id): self._selectedImage = id

    def selectedImage(self): return self._selectedImage

    def getSelectedProcessPipe(self):
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.getSelectedProcessPipe(",  ")")

        res = None
        if self._selectedImage != -1: res= self.processPipes[self._selectedImage]
        return res

    def setImages(self, filenames):
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.setImages(",len(list(copy.deepcopy(filenames))), "images)")

        self.imageFilenames = list(filenames)
        self.imagesMetadata, self.processPipes =  [], [] # reset metadata and processPipes

        self.aestheticsModels = [] # reset aesthetics models

        nbImagePage = controller.GalleryMode.nbRow(self.controller.view.shapeMode)*controller.GalleryMode.nbCol(self.controller.view.shapeMode)
        for f in self.imageFilenames: # load only first page
            self.processPipes.append(None)
        self.controller.updateImages() # update controller to update view
        self.loadPage(0)

    def loadPage(self,nb):
        """loadPage: 
            Args:
                nb(int): page number
            Returns:
        """
        if pref.verbose:  print(" [MODEL] >> ImageGalleryModel.loadPage(",nb,")")
        nbImagePage = controller.GalleryMode.nbRow(self.controller.view.shapeMode)*controller.GalleryMode.nbCol(self.controller.view.shapeMode)
        min_,max_ = (nb*nbImagePage), ((nb+1)*nbImagePage)

        loadThreads = thread.RequestLoadImage(self)

        for i,f in enumerate(self.imageFilenames[min_:max_]): # load only the current page nb
            if not isinstance(self.processPipes[min_+i],hdrCore.processing.ProcessPipe):
                self.controller.parent.statusBar().showMessage("read image: "+f)
                self.controller.parent.statusBar().repaint()
                loadThreads.requestLoad(min_,i, f)
            else:
                self.controller.view.updateImage(i, self.processPipes[min_+i], f)

    def save(self):
        if pref.verbose:  print(" [MODEL] >> ImageGalleryModel.save()")

        for i,p in enumerate(self.processPipes):
            if isinstance(p, hdrCore.processing.ProcessPipe): 
                p.getImage().metadata.metadata['processpipe'] = p.toDict()            
                p.getImage().metadata.save()

    def getFilenamesOfCurrentPage(self):
        minIdx, maxIdx = self.controller.pageIdx()
        return copy.deepcopy(self.imageFilenames[minIdx:maxIdx])

    def getProcessPipeById(self,i):
        return self.processPipes[i]

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AppModel(object):
    """ class for main window model """

    def __init__(self, controller):
        # attributes
        self.controller = controller
        self.directory = pref.getImagePath()
        self.imageFilenames = []
        self.displayHDRProcess = None
        #V5
        self.displayModel = {'scaling':12, 'shape':(2160,3840)}

    def setDirectory(self,path):
        # read directory and return image filename list
        self.directory =path
        pref.setImagePath(path)
        self.imageFilenames = map(lambda x: os.path.join(self.directory,x),
                                  hdrCore.utils.filterlistdir(self.directory,('.jpg','.JPG','.hdr','.HDR')))

        return self.imageFilenames
# ------------------------------------------------------------------------------------------
# --- Class EditImageModel -----------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class EditImageModel(object):
    """
    xxxx xxxx xxxx.
    
    Attributes:
        controller: 
            reference to controller 
        autoPreviewHDR: boolean
        processPipe: processing.ProcessPipe
            current selected ProcessPipe

    Methods:
        __init__(...):
        getProcessPipe(...):
        setProcessPipe(...):
        buildProcessPipe(...):
        autoExpousre(...): called when autoExpouse is clicked
        changeExposure(...): called when exposure value has been changed by user in gui
        getEV(..): 
        changeContrast(...): called when contrast value has been changed by user in gui
        changeToneCurve(...): called when tone-curve values have been changed by user in gui
        changeLightnessMask(...): called when lightness mask parameters have been changed by user in gui
        changeSaturation(...): called when contrast value has been changed by user in gui
        changeColorEditor(...): called when color editor parameters have been changed by user in gui
        changeGeometry(...): called when geometry value has been changed by user in gui    

    """
    def __init__(self,controller):

        self.controller = controller

        self.autoPreviewHDR = False

        # ref to ImageGalleryModel.processPipes[ImageGalleryModel._selectedImage]
        self.processpipe = None 

        # create a RequestCompute
        self.requestCompute  = thread.RequestCompute(self)

    def getProcessPipe(self): return self.processpipe

    def setProcessPipe(self, processPipe): 
        if self.requestCompute.readyToRun:
            self.processpipe = processPipe
            self.requestCompute.setProcessPipe(self.processpipe)
            self.processpipe.compute()
            if self.controller.previewHDR and self.autoPreviewHDR:
                img = self.processpipe.getImage(toneMap = False)
                self.controller.controllerHDR.displayIMG(img)
            return True
        else:
            return False

    @staticmethod
    def buildProcessPipe():
        """
        WARNING: 
            here the process-pipe is built
            initial pipe does not have input image
            initial pipe has processes node according to EditImageView 
        """
        processPipe = hdrCore.processing.ProcessPipe()

        # exposure ---------------------------------------------------------------------------------------------------------
        defaultParameterEV = {'EV': 0}                                              
        idExposureProcessNode = processPipe.append(hdrCore.processing.exposure(), paramDict=None,name="exposure")   
        processPipe.setParameters(idExposureProcessNode, defaultParameterEV)                                        

        # contrast ---------------------------------------------------------------------------------------------------------
        defaultParameterContrast = {'contrast': 0}                                  
        idContrastProcessNode = processPipe.append(hdrCore.processing.contrast(), paramDict=None,  name="contrast") 
        processPipe.setParameters(idContrastProcessNode, defaultParameterContrast)                                  

        #tonecurve ---------------------------------------------------------------------------------------------------------
        defaultParameterYcurve = {'start':[0,0], 
                                  'shadows': [10,10],
                                  'blacks': [30,30], 
                                  'mediums': [50,50], 
                                  'whites': [70,70], 
                                  'highlights': [90,90], 
                                  'end': [100,100]}                         
        idYcurveProcessNode = processPipe.append(hdrCore.processing.Ycurve(), paramDict=None,name="tonecurve")      
        processPipe.setParameters(idYcurveProcessNode, defaultParameterYcurve)   
        
        # masklightness ---------------------------------------------------------------------------------------------------------
        defaultMask = { 'shadows': False, 
                       'blacks': False, 
                       'mediums': False, 
                       'whites': False, 
                       'highlights': False}
        idLightnessMaskProcessNode = processPipe.append(hdrCore.processing.lightnessMask(), paramDict=None, name="lightnessmask")  
        processPipe.setParameters(idLightnessMaskProcessNode, defaultMask)  

        # saturation ---------------------------------------------------------------------------------------------------------
        defaultValue = {'saturation': 0.0,  'method': 'gamma'}
        idSaturationProcessNode = processPipe.append(hdrCore.processing.saturation(), paramDict=None, name="saturation")    
        processPipe.setParameters(idSaturationProcessNode, defaultValue)                     

        # colorEditor0 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor0= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor0ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor0")  
        processPipe.setParameters(idColorEditor0ProcessNode, defaultParameterColorEditor0)

        # colorEditor1 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor1= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor1ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor1")  
        processPipe.setParameters(idColorEditor1ProcessNode, defaultParameterColorEditor1)
        
        # colorEditor2 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor2= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor2ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor2")  
        processPipe.setParameters(idColorEditor2ProcessNode, defaultParameterColorEditor2)
        
        # colorEditor3 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor3= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor3ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor3")  
        processPipe.setParameters(idColorEditor3ProcessNode, defaultParameterColorEditor3)
        
        # colorEditor4 ---------------------------------------------------------------------------------------------------------
        defaultParameterColorEditor4= {'selection': {'lightness': (0,100),'chroma': (0,100),'hue':(0,360)},  
                                       'edit': {'hue': 0.0, 'exposure':0.0, 'contrast':0.0,'saturation':0.0}, 
                                       'mask': False}        
        idColorEditor4ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor4")  
        processPipe.setParameters(idColorEditor4ProcessNode, defaultParameterColorEditor4)

        # geometry ---------------------------------------------------------------------------------------------------------
        defaultValue = { 'ratio': (16,9), 'up': 0,'rotation': 0.0}
        idGeometryNode = processPipe.append(hdrCore.processing.geometry(), paramDict=None, name="geometry")    
        processPipe.setParameters(idGeometryNode, defaultValue)
        # ------------ --------------------------------------------------------------------------------------------------------- 

        return processPipe

    def autoExposure(self):
        if pref.verbose:  print(" [MODEL] >> EditImageModel.autoExposure(",")")

        id = self.processpipe.getProcessNodeByName("exposure")
        exposureProcess = self.processpipe.processNodes[id].process
        img= self.processpipe.getInputImage()
        EV = exposureProcess.auto(img)
        self.processpipe.setParameters(id,EV)

        self.processpipe.compute()

        if self.controller.previewHDR and self.autoPreviewHDR:
            img = self.processpipe.getImage(toneMap = False)
            self.controller.controllerHDR.displayIMG(img)

        return self.processpipe.getImage()

    def changeExposure(self,value):
        """
        changeExposure: XXX

        Args:
            XXX
        Returns:
            XXX
        """

        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeExposure(",value,")")

        id = self.processpipe.getProcessNodeByName("exposure")
        self.requestCompute.requestCompute(id,{'EV': value})

    def getEV(self):
        if pref.verbose:  print(" [MODEL] >> EditImageModel.getEV(",")")
        id = self.processpipe.getProcessNodeByName("exposure")
        return self.processpipe.getParameters(id)

    def changeContrast(self,value):
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeContrast(",value,")")

        id = self.processpipe.getProcessNodeByName("contrast")
        self.requestCompute.requestCompute(id,{'contrast': value})

    def changeToneCurve(self,controlPoints):
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeToneCurve(",")")

        id = self.processpipe.getProcessNodeByName("tonecurve")
        self.requestCompute.requestCompute(id,controlPoints)

    def changeLightnessMask(self, maskValues):
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeLightnessMask(",maskValues,")")

        id = self.processpipe.getProcessNodeByName("lightnessmask")
        self.requestCompute.requestCompute(id,maskValues)

    def changeSaturation(self,value):
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeSaturation(",value,")")

        id = self.processpipe.getProcessNodeByName("saturation")
        self.requestCompute.requestCompute(id,{'saturation': value, 'method': 'gamma'})

    def changeColorEditor(self, values, idName):
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeColorEditor(",values,")")

        id = self.processpipe.getProcessNodeByName(idName)
        self.requestCompute.requestCompute(id,values)

    def changeGeometry(self, values):
        if pref.verbose:  print(" [MODEL] >> EditImageModel.changeGeometry(",values,")")

        id = self.processpipe.getProcessNodeByName("geometry")
        self.requestCompute.requestCompute(id,values)

    def updateImage(self, imgTM):
        self.controller.updateImage(imgTM)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceSliderModel():
    def __init__(self, controller, value):
        self.controller = controller
        self.value = value
    def setValue(self, value): self.value =  value
    def toDict(self): return {'value': self.value}
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ToneCurveModel():
    """

        +-------+-------+-------+-------+-------+                             [o]
        |       |       |       |       |       |                              ^
        |       |       |       |       |   o   |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |       |       |       |   o   |       |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |       |       |   o   |       |       |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |       |   o   |       |       |       |                              |
        |       |       |       |       |       |                              |
        +-------+-------+-------+-------+-------+                              |
        |       |       |       |       |       |                              |
        |   o   |       |       |       |       |                              |
        |       |       |       |       |       |                              |
       [o]-------+-------+-------+-------+-------+-----------------------------+ 
  zeros ^ shadows  black   medium  white  highlights                          200



    """
    def __init__(self):
        if pref.verbose: print(" [MODEL] >> ToneCourveModel.__init__()")

        #  start
        self.control = {'start':[0.0,0.0], 'shadows': [10.0,10.0], 'blacks': [30.0,30.0], 'mediums': [50.0,50.0], 'whites': [70.0,70.0], 'highlights': [90.0,90.0], 'end': [100.0,100.0]}
        self.default = {'start':[0.0,0.0], 'shadows': [10.0,10.0], 'blacks': [30.0,30.0], 'mediums': [50.0,50.0], 'whites': [70.0,70.0], 'highlights': [90.0,90.0], 'end': [100.0,100.0]}

        self.curve =    BSpline.Curve()
        self.curve.degree = 2
        self.points =None

    def evaluate(self):
        if pref.verbose: print(" [MODEL] >> ToneCurveModel.evaluate(",")")

        #self.curve.ctrlpts = copy.deepcopy([self.control['start'],self.control['shadows'],self.control['blacks'],self.control['mediums'], self.control['whites'], self.control['highlights'], self.control['end']])
        self.curve.ctrlpts = copy.deepcopy([self.control['start'],self.control['shadows'],self.control['blacks'],self.control['mediums'], self.control['whites'], self.control['highlights'], [200, self.control['end'][1]]])
        # auto-generate knot vector
        self.curve.knotvector = utilities.generate_knot_vector(self.curve.degree, len(self.curve.ctrlpts))
        # evaluate curve and get points
        self.points = np.asarray(self.curve.evalpts)

        return self.points

    def setValue(self, key, value, autoScale=False):
        if pref.verbose: print(" [MODEL] >> ToneCourveModel.setValue(",key,", ",value,", autoScale=",autoScale,")")
        value = float(value)
        # check key
        if key in self.control.keys():
            # transform in list
            listKeys = list(self.control.keys())
            listValues = np.asarray(list(self.control.values()))
            index = listKeys.index(key)

            if (listValues[:index,1] <= value).all() and (value <= listValues[index+1:,1]).all():
                # can change
                oldValue = self.control[listKeys[index]]
                self.control[listKeys[index]] = [oldValue[0],value]            
            elif not (value <= listValues[index+1:,1]).all():
                if autoScale:
                    minValue = min(listValues[index:,1])
                    maxValue = listValues[-1:,1]
                    u = (listValues[index+1:,1] -minValue)/(maxValue-minValue)

                    newValues = value*(1- u)+ u*maxValue
                    for i,v in enumerate(newValues):
                        oldValue = self.control[listKeys[i+index+1]]
                        self.control[listKeys[i+index+1]] = [oldValue[0],np.round(v)]
                else:
                    # not autoScale, set to minValue
                    oldValue = self.control[listKeys[index]]
                    minValue = min(listValues[index+1:,1])
                    self.control[listKeys[index]] = [oldValue[0],minValue]
            elif not (listValues[:index,1] <= value).all():
                if autoScale:
                    minValue = listValues[0,1]
                    maxValue = max(listValues[:index,1])
                    u = (listValues[:index,1] -minValue)/(maxValue-minValue)
                    newValues = minValue*(1- u)+ u*value
                    for i,v in enumerate(newValues):
                        oldValue = self.control[listKeys[i]]
                        self.control[listKeys[i]] = [oldValue[0],np.round(v)]
                else:
                    # not autoScale, set to maxValue
                    oldValue = self.control[listKeys[index]]
                    maxValue =  max(listValues[:index,1])
                    self.control[listKeys[index]] = [oldValue[0],maxValue]

        return self.control

    def setValues(self, controlPointsDict):
        self.control = copy.deepcopy(controlPointsDict)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class LightnessMaskModel():
    def __init__(self, _controller):
        self.controller = _controller

        self.masks = {'shadows': False, 'blacks':False, 'mediums':False, 'whites':False, 'highlights':False}

    def maskChange(self, key, on_off):
        if key in self.masks.keys(): self.masks[key] = on_off
        return copy.deepcopy(self.masks)

    def setValues(self, values): self.masks = copy.deepcopy(values)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageInfoModel(object):

    def __init__(self,controller):
        self.controller = controller

        # ref to ImageGalleryModel.processPipes[ImageGalleryModel._selectedImage]
        self.processPipe = None 

    def getProcessPipe(self): return self.processPipe

    def setProcessPipe(self, processPipe):  self.processPipe = processPipe

    def changeMeta(self,tagGroup,tag, on_off): 
        if pref.verbose:  print(" [MODEL] >> ImageInfoModel.changeMeta(",tagGroup,",",tag,",", on_off,")")

        if isinstance(self.processPipe, hdrCore.processing.ProcessPipe):
            tagRootName = self.processPipe.getImage().metadata.otherTags.getTagsRootName()
            tags = copy.deepcopy(self.processPipe.getImage().metadata.metadata[tagRootName])
            found, updatedMeta = False, []
            for tt in tags:
                if tagGroup in tt.keys():
                    if tag in tt[tagGroup].keys():
                        found = True
                        tt[tagGroup][tag] = on_off 
                updatedMeta.append(copy.deepcopy(tt))
            self.processPipe.updateUserMeta(tagRootName,updatedMeta)
            if pref.verbose: 
                print(" [MODEL] >> ImageInfoModel.changeUseCase(",")")
                for tt in updatedMeta: print(tt)     
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class CurveControlModel(object): pass
# ------------------------------------------------------------------------------------------
# ---- HDRviewerModel ----------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class HDRviewerModel(object):
    def __init__(self,_controller):
        if pref.verbose: print(" [MODEL] >> HDRviewerModel.__init__(",")")

        self.controller = _controller

        # current image
        self.currentIMG = None

        self.displayModel = pref.getHDRdisplays()

    def scaling(self): 
        if pref.verbose: print(f" [MODEL] >> HDRviewerModel.scaling():{self.displayModel['scaling']}")
        return self.displayModel['scaling']

    def shape(self): 
        if pref.verbose: print(f" [MODEL] >> HDRviewerModel.shape():{self.displayModel['shape']}")        
        return self.displayModel['shape']
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class LchColorSelectorModel(object):
    def __init__(self, _controller):
        self.controller = _controller

        self.lightnessSelection =   (0,100)     # min, max
        self.chromaSelection =      (0,100)     # min, max
        self.hueSelection =         (0,360)     # min, max

        self.exposure =     0.0
        self.hueShift =     0.0
        self.contrast =     0.0
        self.saturation =   0.0

        self.mask =         False

        # -------------
        self.default = {
            "selection": {"lightness": [ 0, 100 ],"chroma": [ 0, 100 ],"hue": [ 0, 360 ]},
            "edit": {"hue": 0,"exposure": 0,"contrast": 0,"saturation": 0},
            "mask": False}
        # -------------


    def setHueSelection(self, hMin,hMax):
        self.hueSelection = hMin,hMax
        return self.getValues()

    def setChromaSelection(self, cMin, cMax):              
        self.chromaSelection = cMin, cMax
        return self.getValues()

    def setLightnessSelection(self, lMin, lMax):              
        self.lightnessSelection = lMin, lMax
        return self.getValues()

    def setExposure(self,ev):
        self.exposure = ev
        return self.getValues()

    def setHueShift(self,hs):
        self.hueShift = hs
        return self.getValues()

    def setContrast(self, contrast):
        self.contrast = contrast
        return self.getValues()

    def setSaturation(self, saturation): 
        self.saturation = saturation
        return self.getValues()

    def setMask(self, value): 
        self.mask = value
        return self.getValues()

    def getValues(self):
        return {
            'selection':    {'lightness': self.lightnessSelection ,'chroma': self.chromaSelection,'hue':self.hueSelection}, 
            'edit':         {'hue':self.hueShift,'exposure':self.exposure,'contrast':self.contrast,'saturation':self.saturation}, 
            'mask':         self.mask
            }

    def setValues(self, values):
        self.lightnessSelection =   values['selection']['lightness']    if 'lightness' in values['selection'].keys() else (0,100)
        self.chromaSelection =      values['selection']['chroma']       if 'chroma' in values['selection'].keys() else (0,100)
        self.hueSelection =         values['selection']['hue']          if 'hue' in values['selection'].keys() else (0,360)

        self.exposure =     values['edit']['exposure']      if 'exposure' in values['edit'].keys() else 0
        self.hueShift =     values['edit']['hue']           if 'hue' in values['edit'].keys() else 0
        self.contrast =     values['edit']['contrast']      if 'contrast' in values['edit'].keys() else 0
        self.saturation =   values['edit']['saturation']    if 'saturation' in values['edit'].keys() else 0

        self.mask =         values['mask']  if 'mask' in values.keys() else False
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class GeometryModel(object):
    def __init__(self, _controller):
        self.controller = _controller

        self.ratio =    (16,9)    
        self.up =       0.0
        self.rotation = 0.0 

    def setCroppingVerticalAdjustement(self,up):
        self.up = up
        return self.getValues()

    def setRotation(self, rotation):              
        self.rotation = rotation
        return self.getValues()

    def getValues(self):
        return { 'ratio': self.ratio, 'up': self.up,'rotation': self.rotation}

    def setValues(self, values):
        self.ratio =    values['ratio']     if 'ratio'      in values.keys() else (16,9)
        self.up =       values['up']        if 'up'         in values.keys() else 0.0
        self.rotation = values['rotation']  if 'rotation'   in values.keys() else 0.0
# ------------------------------------------------------------------------------------------
# ---- Class AestheticsImageModel ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageAestheticsModel:
    """class ImageAesthetics: encapsulates color palette (and related parameters), convexHull composition (and related parameters), etc.

        Attributes:
            parent (guiQt.controller.ImageAestheticsController): controller
            processPipe (hdrCore.processing.ProcessPipe): current selected process-pipe

        Methods:


    """
    def __init__(self, parent):
        if pref.verbose: print(" [MODEL] >> ImageAestheticsModel.__init__(",")")
        
        self.parent = parent

        # processPipeHasChanged
        self.requireUpdate = True


        # ref to ImageGalleryModel.processPipes[ImageGalleryModel._selectedImage]
        self.processPipe = None 

        # color palette
        self.colorPalette = hdrCore.aesthetics.Palette('defaultLab5',
                                                       np.linspace([0,0,0],[100,0,0],5),
                                                       hdrCore.image.ColorSpace.build('Lab'), 
                                                       hdrCore.image.imageType.SDR)
    # ------------------------------------------------------------------------------------------
    def getProcessPipe(self): return self.processPipe
    # ------------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe):  
        if pref.verbose: print(" [MODEL] >> ImageAestheticsModel.setProcessPipe(",")")

        if processPipe != self.processPipe:
        
            self.processPipe = processPipe
            self.requireUpdate = True

        if self.requireUpdate:

            self.colorPalette = hdrCore.aesthetics.Palette.build(self.processPipe)
            # COMPUTE IMAGE OF PALETTE
            paletteIMG = self.colorPalette.createImageOfPalette()

            self.endComputing()

        else: pass
    # ------------------------------------------------------------------------------------------
    def endComputing(self):
        self.requireUpdate = False
    # ------------------------------------------------------------------------------------------
    def getPaletteImage(self):
        return self.colorPalette.createImageOfPalette()
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ColorEditorsAutoModel:
    def __init__(self,_controller, processStepName,nbColors, removeBlack= True):
        self.controller = _controller
        self.processStepId = processStepName
        self.nbColors = nbColors
        self.removeBlack = removeBlack

    def compute(self):
        # get image according to processId
        processPipe = self.controller.parent.controller.getProcessPipe()
        if processPipe != None:
            image_ = processPipe.processNodes[processPipe.getProcessNodeByName(self.processStepId)].outputImage

            if image_.colorSpace.name == 'Lch':
                LchPixels = image_.colorData
            elif image_.colorSpace.name == 'sRGB':
                if image_.linear: 
                    colorLab = hdrCore.processing.sRGB_to_Lab(image_.colorData, apply_cctf_decoding=False)
                    LchPixels = colour.Lab_to_LCHab(colorLab)
                else:
                    colorLab = hdrCore.processing.sRGB_to_Lab(image_.colorData, apply_cctf_decoding=True)
                    LchPixels = colour.Lab_to_LCHab(colorLab)

            # to Lab then to Vector
            LabPixels = colour.LCHab_to_Lab(LchPixels)
            LabPixelsVector = hdrCore.utils.ndarray2vector(LabPixels)

            # k-means: nb cluster = nbColors + 1
            kmeans_cluster_Lab = sklearn.cluster.KMeans(n_clusters=self.nbColors+1)
            kmeans_cluster_Lab.fit(LabPixelsVector)
            cluster_centers_Lab = kmeans_cluster_Lab.cluster_centers_
            cluster_labels_Lab = kmeans_cluster_Lab.labels_
                
            # remove darkness one
            idxLmin = np.argmin(cluster_centers_Lab[:,0])                           # idx of darkness
            cluster_centers_Lab = np.delete(cluster_centers_Lab, idxLmin, axis=0)   # remove min from cluster_centers_Lab

            # go to Lch
            cluster_centers_Lch = colour.Lab_to_LCHab(cluster_centers_Lab) 

            # sort cluster by hue
            cluster_centersIdx = np.argsort(cluster_centers_Lch[:,2])

            dictValuesList = []
            for j in range(len(cluster_centersIdx)):
                i = cluster_centersIdx[j]
                if j==0: Hmin = 0
                else: Hmin = 0.5*(cluster_centers_Lch[cluster_centersIdx[j-1]][2] + cluster_centers_Lch[cluster_centersIdx[j]][2])
                if j == len(cluster_centersIdx)-1: Hmax = 360
                else: Hmax = 0.5*(cluster_centers_Lch[cluster_centersIdx[j]][2] + cluster_centers_Lch[cluster_centersIdx[j+1]][2])

                Cmin = max(0, cluster_centers_Lch[cluster_centersIdx[j]][1]-25) 
                Cmax = min(100, cluster_centers_Lch[cluster_centersIdx[j]][1]+25) 

                Lmin = max(0, cluster_centers_Lch[cluster_centersIdx[j]][0]-25) 
                Lmax = min(100, cluster_centers_Lch[cluster_centersIdx[j]][0]+25) 

                dictSegment =  {
                    "selection": {
                        "lightness":    [ Lmin,  Lmax],
                        "chroma":       [ Cmin,  Cmax],
                        "hue":          [ Hmin,  Hmax ]},
                    "edit": {"hue": 0,"exposure": 0,"contrast": 0,"saturation": 0},
                    "mask": False}
                dictValuesList.append(dictSegment)

            return dictValuesList
# ------------------------------------------------------------------------------------------






