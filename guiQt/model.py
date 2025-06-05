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
    """Image GUI model"""

    def __init__(self, controller):
        self.controller = controller
        self.image = None # numpy array or hdrCore.image.Image

    def setImage(self, image):
        """Set the image."""
        self.image = image

    def getColorData(self):
        """Get the color data of the image.

        Returns:
            numpy.ndarray or hdrCore.image.Image.colorData: The color data of the image.
        """
        if isinstance(self.image, np.ndarray):
            return self.image
        elif isinstance(self.image, hdrCore.image.Image):
            return self.image.colorData

class ImageGalleryModel:
    """ImageGalleryModel: Management of image gallery

    Attributes:
        controller (ImageGalleryController): Parent of self
        imageFilenames (list[str]): List of image filenames
        processPipes (list[hdrCore.processing.ProcessPipe]): List of process-pipes associated with images
        _selectedImage (int): Index of current (selected) process-pipe
        aestheticsModels (list[hdrCore.aesthetics.MultidimensionalImageAestheticsModel])

    Methods:
        setSelectedImage: Set the selected image index
        selectedImage: Get the selected image index
        getSelectedProcessPipe: Get the process pipe of the selected image
        setImages: Set the list of image filenames
        loadPage: Load a specific page of images
        save: Save the current state
        getFilenamesOfCurrentPage: Get the filenames of the current page
        getProcessPipeById: Get the process pipe by its ID
    """

    def __init__(self, _controller):
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.__init__()")
        self.controller = _controller
        self.imageFilenames = []
        self.processPipes = []
        self._selectedImage = -1
        self.aesthetics = []

    def setSelectedImage(self, id):
        """Set the selected image index.

        Args:
            id (int): Index of the selected image
        """
        self._selectedImage = id

    def selectedImage(self):
        """Get the selected image index.

        Returns:
            int: Index of the selected image
        """
        return self._selectedImage

    def getSelectedProcessPipe(self):
        """Get the process pipe of the selected image.

        Returns:
            hdrCore.processing.ProcessPipe or None: Process pipe of the selected image
        """
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.getSelectedProcessPipe()")
        res = None
        if self._selectedImage != -1: res = self.processPipes[self._selectedImage]
        return res

    def setImages(self, filenames):
        """Set the list of image filenames.

        Args:
            filenames (list[str]): List of image filenames
        """
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.setImages(", len(list(copy.deepcopy(filenames))), "images)")
        self.imageFilenames = list(filenames)
        self.imagesMetadata, self.processPipes = [], [] # reset metadata and processPipes
        self.aestheticsModels = [] # reset aesthetics models
        nbImagePage = controller.GalleryMode.nbRow(self.controller.view.shapeMode) * controller.GalleryMode.nbCol(self.controller.view.shapeMode)
        for f in self.imageFilenames: # load only first page
            self.processPipes.append(None)
        self.controller.updateImages() # update controller to update view
        self.loadPage(0)

    def loadPage(self, nb):
        """Load a specific page of images.

        Args:
            nb (int): Page number
        """
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.loadPage(", nb, ")")
        nbImagePage = controller.GalleryMode.nbRow(self.controller.view.shapeMode) * controller.GalleryMode.nbCol(self.controller.view.shapeMode)
        min_, max_ = (nb * nbImagePage), ((nb + 1) * nbImagePage)
        loadThreads = thread.RequestLoadImage(self)
        for i, f in enumerate(self.imageFilenames[min_:max_]): # load only the current page nb
            if not isinstance(self.processPipes[min_ + i], hdrCore.processing.ProcessPipe):
                self.controller.parent.statusBar().showMessage("read image: " + f)
                self.controller.parent.statusBar().repaint()
                loadThreads.requestLoad(min_, i, f)
            else:
                self.controller.view.updateImage(i, self.processPipes[min_ + i], f)

    def save(self):
        """Save the current state."""
        if pref.verbose: print(" [MODEL] >> ImageGalleryModel.save()")
        for i, p in enumerate(self.processPipes):
            if isinstance(p, hdrCore.processing.ProcessPipe):
                p.getImage().metadata.metadata['processpipe'] = p.toDict()
                p.getImage().metadata.save()

    def getFilenamesOfCurrentPage(self):
        """Get the filenames of the current page.

        Returns:
            list[str]: Filenames of the current page
        """
        minIdx, maxIdx = self.controller.pageIdx()
        return copy.deepcopy(self.imageFilenames[minIdx:maxIdx])

    def getProcessPipeById(self, i):
        """Get the process pipe by its ID.

        Args:
            i (int): ID of the process pipe

        Returns:
            hdrCore.processing.ProcessPipe: Process pipe with the given ID
        """
        return self.processPipes[i]

class AppModel(object):
    """Class for main window model"""

    def __init__(self, controller):
        """Initialize the AppModel.

        Args:
            controller: Reference to the controller
        """
        self.controller = controller
        self.directory = pref.getImagePath()
        self.imageFilenames = []
        self.displayHDRProcess = None
        self.displayModel = {'scaling': 12, 'shape': (2160, 3840)}

    def setDirectory(self, path):
        """Set the directory and read image filenames.

        Args:
            path (str): Path to the directory

        Returns:
            list[str]: List of image filenames
        """
        self.directory = path
        pref.setImagePath(path)
        self.imageFilenames = map(lambda x: os.path.join(self.directory, x),
                                  hdrCore.utils.filterlistdir(self.directory, ('.jpg', '.JPG', '.hdr', '.HDR')))
        return self.imageFilenames

class EditImageModel(object):
    """Model for editing images.

    Attributes:
        controller: Reference to the controller
        autoPreviewHDR (bool): Boolean for auto preview HDR
        processPipe (processing.ProcessPipe): Current selected ProcessPipe

    Methods:
        __init__(...): Initialize the EditImageModel
        getProcessPipe(...): Get the current process pipe
        setProcessPipe(...): Set the current process pipe
        buildProcessPipe(...): Build a new process pipe
        autoExposure(...): Called when auto exposure is clicked
        changeExposure(...): Called when exposure value has been changed by user in GUI
        getEV(...): Get the exposure value
        changeContrast(...): Called when contrast value has been changed by user in GUI
        changeToneCurve(...): Called when tone-curve values have been changed by user in GUI
        changeLightnessMask(...): Called when lightness mask parameters have been changed by user in GUI
        changeSaturation(...): Called when saturation value has been changed by user in GUI
        changeColorEditor(...): Called when color editor parameters have been changed by user in GUI
        changeGeometry(...): Called when geometry value has been changed by user in GUI
    """

    def __init__(self, controller):
        """Initialize the EditImageModel.

        Args:
            controller: Reference to the controller
        """
        self.controller = controller
        self.autoPreviewHDR = False
        self.processpipe = None
        self.requestCompute = thread.RequestCompute(self)

    def getProcessPipe(self):
        """Get the current process pipe.

        Returns:
            processing.ProcessPipe: Current process pipe
        """
        return self.processpipe

    def setProcessPipe(self, processPipe):
        """Set the current process pipe.

        Args:
            processPipe (processing.ProcessPipe): Process pipe to set

        Returns:
            bool: True if the process pipe was set successfully, False otherwise
        """
        if self.requestCompute.readyToRun:
            self.processpipe = processPipe
            self.requestCompute.setProcessPipe(self.processpipe)
            self.processpipe.compute()
            if self.controller.previewHDR and self.autoPreviewHDR:
                img = self.processpipe.getImage(toneMap=False)
                self.controller.controllerHDR.displayIMG(img)
            return True
        else:
            return False

    @staticmethod
    def buildProcessPipe():
        """Build a new process pipe.

        Returns:
            processing.ProcessPipe: Newly built process pipe
        """
        processPipe = hdrCore.processing.ProcessPipe()

        # Exposure
        defaultParameterEV = {'EV': 0}
        idExposureProcessNode = processPipe.append(hdrCore.processing.exposure(), paramDict=None, name="exposure")
        processPipe.setParameters(idExposureProcessNode, defaultParameterEV)

        # Contrast
        defaultParameterContrast = {'contrast': 0}
        idContrastProcessNode = processPipe.append(hdrCore.processing.contrast(), paramDict=None, name="contrast")
        processPipe.setParameters(idContrastProcessNode, defaultParameterContrast)

        # Tone curve
        defaultParameterYcurve = {'start': [0, 0],
                                  'shadows': [10, 10],
                                  'blacks': [30, 30],
                                  'mediums': [50, 50],
                                  'whites': [70, 70],
                                  'highlights': [90, 90],
                                  'end': [100, 100]}
        idYcurveProcessNode = processPipe.append(hdrCore.processing.Ycurve(), paramDict=None, name="tonecurve")
        processPipe.setParameters(idYcurveProcessNode, defaultParameterYcurve)

        # Lightness mask
        defaultMask = {'shadows': False,
                       'blacks': False,
                       'mediums': False,
                       'whites': False,
                       'highlights': False}
        idLightnessMaskProcessNode = processPipe.append(hdrCore.processing.lightnessMask(), paramDict=None, name="lightnessmask")
        processPipe.setParameters(idLightnessMaskProcessNode, defaultMask)

        # Saturation
        defaultValue = {'saturation': 0.0, 'method': 'gamma'}
        idSaturationProcessNode = processPipe.append(hdrCore.processing.saturation(), paramDict=None, name="saturation")
        processPipe.setParameters(idSaturationProcessNode, defaultValue)

        # Color editor 0
        defaultParameterColorEditor0 = {'selection': {'lightness': (0, 100), 'chroma': (0, 100), 'hue': (0, 360)},
                                       'edit': {'hue': 0.0, 'exposure': 0.0, 'contrast': 0.0, 'saturation': 0.0},
                                       'mask': False}
        idColorEditor0ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor0")
        processPipe.setParameters(idColorEditor0ProcessNode, defaultParameterColorEditor0)

        # Color editor 1
        defaultParameterColorEditor1 = {'selection': {'lightness': (0, 100), 'chroma': (0, 100), 'hue': (0, 360)},
                                       'edit': {'hue': 0.0, 'exposure': 0.0, 'contrast': 0.0, 'saturation': 0.0},
                                       'mask': False}
        idColorEditor1ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor1")
        processPipe.setParameters(idColorEditor1ProcessNode, defaultParameterColorEditor1)

        # Color editor 2
        defaultParameterColorEditor2 = {'selection': {'lightness': (0, 100), 'chroma': (0, 100), 'hue': (0, 360)},
                                       'edit': {'hue': 0.0, 'exposure': 0.0, 'contrast': 0.0, 'saturation': 0.0},
                                       'mask': False}
        idColorEditor2ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor2")
        processPipe.setParameters(idColorEditor2ProcessNode, defaultParameterColorEditor2)

        # Color editor 3
        defaultParameterColorEditor3 = {'selection': {'lightness': (0, 100), 'chroma': (0, 100), 'hue': (0, 360)},
                                       'edit': {'hue': 0.0, 'exposure': 0.0, 'contrast': 0.0, 'saturation': 0.0},
                                       'mask': False}
        idColorEditor3ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor3")
        processPipe.setParameters(idColorEditor3ProcessNode, defaultParameterColorEditor3)

        # Color editor 4
        defaultParameterColorEditor4 = {'selection': {'lightness': (0, 100), 'chroma': (0, 100), 'hue': (0, 360)},
                                       'edit': {'hue': 0.0, 'exposure': 0.0, 'contrast': 0.0, 'saturation': 0.0},
                                       'mask': False}
        idColorEditor4ProcessNode = processPipe.append(hdrCore.processing.colorEditor(), paramDict=None, name="colorEditor4")
        processPipe.setParameters(idColorEditor4ProcessNode, defaultParameterColorEditor4)

        # Geometry
        defaultValue = {'ratio': (16, 9), 'up': 0, 'rotation': 0.0}
        idGeometryNode = processPipe.append(hdrCore.processing.geometry(), paramDict=None, name="geometry")
        processPipe.setParameters(idGeometryNode, defaultValue)

        return processPipe

    def autoExposure(self):
        """Auto exposure adjustment.

        Returns:
            Image: Processed image
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.autoExposure()")
        id = self.processpipe.getProcessNodeByName("exposure")
        exposureProcess = self.processpipe.processNodes[id].process
        img = self.processpipe.getInputImage()
        EV = exposureProcess.auto(img)
        self.processpipe.setParameters(id, EV)
        self.processpipe.compute()
        if self.controller.previewHDR and self.autoPreviewHDR:
            img = self.processpipe.getImage(toneMap=False)
            self.controller.controllerHDR.displayIMG(img)
        return self.processpipe.getImage()

    def changeExposure(self, value):
        """Change the exposure value.

        Args:
            value: New exposure value
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeExposure(", value, ")")
        id = self.processpipe.getProcessNodeByName("exposure")
        self.requestCompute.requestCompute(id, {'EV': value})

    def getEV(self):
        """Get the exposure value.

        Returns:
            dict: Exposure parameters
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.getEV()")
        id = self.processpipe.getProcessNodeByName("exposure")
        return self.processpipe.getParameters(id)

    def changeContrast(self, value):
        """Change the contrast value.

        Args:
            value: New contrast value
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeContrast(", value, ")")
        id = self.processpipe.getProcessNodeByName("contrast")
        self.requestCompute.requestCompute(id, {'contrast': value})

    def changeToneCurve(self, controlPoints):
        """Change the tone curve values.

        Args:
            controlPoints: New control points for the tone curve
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeToneCurve()")
        id = self.processpipe.getProcessNodeByName("tonecurve")
        self.requestCompute.requestCompute(id, controlPoints)

    def changeLightnessMask(self, maskValues):
        """Change the lightness mask parameters.

        Args:
            maskValues: New mask values
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeLightnessMask(", maskValues, ")")
        id = self.processpipe.getProcessNodeByName("lightnessmask")
        self.requestCompute.requestCompute(id, maskValues)

    def changeSaturation(self, value):
        """Change the saturation value.

        Args:
            value: New saturation value
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeSaturation(", value, ")")
        id = self.processpipe.getProcessNodeByName("saturation")
        self.requestCompute.requestCompute(id, {'saturation': value, 'method': 'gamma'})

    def changeColorEditor(self, values, idName):
        """Change the color editor parameters.

        Args:
            values: New color editor values
            idName: Name of the color editor node
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeColorEditor(", values, ")")
        id = self.processpipe.getProcessNodeByName(idName)
        self.requestCompute.requestCompute(id, values)

    def changeGeometry(self, values):
        """Change the geometry values.

        Args:
            values: New geometry values
        """
        if pref.verbose: print(" [MODEL] >> EditImageModel.changeGeometry(", values, ")")
        id = self.processpipe.getProcessNodeByName("geometry")
        self.requestCompute.requestCompute(id, values)

    def updateImage(self, imgTM):
        """Update the image.

        Args:
            imgTM: Tone-mapped image
        """
        self.controller.updateImage(imgTM)

class AdvanceSliderModel():
    """Model for an advanced slider."""

    def __init__(self, controller, value):
        """Initialize the AdvanceSliderModel.

        Args:
            controller: Reference to the controller
            value: Initial value of the slider
        """
        self.controller = controller
        self.value = value

    def setValue(self, value):
        """Set the value of the slider.

        Args:
            value: New value of the slider
        """
        self.value = value

    def toDict(self):
        """Convert the slider value to a dictionary.

        Returns:
            dict: Dictionary containing the slider value
        """
        return {'value': self.value}

class ToneCurveModel():
    """Model for tone curve adjustments.

    Control points diagram:
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
        """Initialize the ToneCurveModel."""
        if pref.verbose: print(" [MODEL] >> ToneCurveModel.__init__()")
        self.control = {'start': [0.0, 0.0], 'shadows': [10.0, 10.0], 'blacks': [30.0, 30.0], 'mediums': [50.0, 50.0], 'whites': [70.0, 70.0], 'highlights': [90.0, 90.0], 'end': [100.0, 100.0]}
        self.default = {'start': [0.0, 0.0], 'shadows': [10.0, 10.0], 'blacks': [30.0, 30.0], 'mediums': [50.0, 50.0], 'whites': [70.0, 70.0], 'highlights': [90.0, 90.0], 'end': [100.0, 100.0]}
        self.curve = BSpline.Curve()
        self.curve.degree = 2
        self.points = None

    def evaluate(self):
        """Evaluate the tone curve and get points.

        Returns:
            numpy.ndarray: Points of the evaluated curve
        """
        if pref.verbose: print(" [MODEL] >> ToneCurveModel.evaluate()")
        self.curve.ctrlpts = copy.deepcopy([self.control['start'], self.control['shadows'], self.control['blacks'], self.control['mediums'], self.control['whites'], self.control['highlights'], [200, self.control['end'][1]]])
        self.curve.knotvector = utilities.generate_knot_vector(self.curve.degree, len(self.curve.ctrlpts))
        self.points = np.asarray(self.curve.evalpts)
        return self.points

    def setValue(self, key, value, autoScale=False):
        """Set the value of a control point.

        Args:
            key: Key of the control point
            value: New value of the control point
            autoScale (bool): Whether to auto-scale the curve

        Returns:
            dict: Updated control points
        """
        if pref.verbose: print(" [MODEL] >> ToneCurveModel.setValue(", key, ", ", value, ", autoScale=", autoScale, ")")
        value = float(value)
        if key in self.control.keys():
            listKeys = list(self.control.keys())
            listValues = np.asarray(list(self.control.values()))
            index = listKeys.index(key)
            if (listValues[:index, 1] <= value).all() and (value <= listValues[index + 1:, 1]).all():
                oldValue = self.control[listKeys[index]]
                self.control[listKeys[index]] = [oldValue[0], value]
            elif not (value <= listValues[index + 1:, 1]).all():
                if autoScale:
                    minValue = min(listValues[index:, 1])
                    maxValue = listValues[-1:, 1]
                    u = (listValues[index + 1:, 1] - minValue) / (maxValue - minValue)
                    newValues = value * (1 - u) + u * maxValue
                    for i, v in enumerate(newValues):
                        oldValue = self.control[listKeys[i + index + 1]]
                        self.control[listKeys[i + index + 1]] = [oldValue[0], np.round(v)]
                else:
                    oldValue = self.control[listKeys[index]]
                    minValue = min(listValues[index + 1:, 1])
                    self.control[listKeys[index]] = [oldValue[0], minValue]
            elif not (listValues[:index, 1] <= value).all():
                if autoScale:
                    minValue = listValues[0, 1]
                    maxValue = max(listValues[:index, 1])
                    u = (listValues[:index, 1] - minValue) / (maxValue - minValue)
                    newValues = minValue * (1 - u) + u * value
                    for i, v in enumerate(newValues):
                        oldValue = self.control[listKeys[i]]
                        self.control[listKeys[i]] = [oldValue[0], np.round(v)]
                else:
                    oldValue = self.control[listKeys[index]]
                    maxValue = max(listValues[:index, 1])
                    self.control[listKeys[index]] = [oldValue[0], maxValue]
        return self.control

    def setValues(self, controlPointsDict):
        """Set the values of the control points.

        Args:
            controlPointsDict (dict): Dictionary of control points
        """
        self.control = copy.deepcopy(controlPointsDict)

class LightnessMaskModel():
    """Model for lightness mask adjustments."""

    def __init__(self, _controller):
        """Initialize the LightnessMaskModel.

        Args:
            _controller: Reference to the controller
        """
        self.controller = _controller
        self.masks = {'shadows': False, 'blacks': False, 'mediums': False, 'whites': False, 'highlights': False}

    def maskChange(self, key, on_off):
        """Change the mask value.

        Args:
            key: Key of the mask
            on_off (bool): New value of the mask

        Returns:
            dict: Updated masks
        """
        if key in self.masks.keys(): self.masks[key] = on_off
        return copy.deepcopy(self.masks)

    def setValues(self, values):
        """Set the values of the masks.

        Args:
            values (dict): Dictionary of mask values
        """
        self.masks = copy.deepcopy(values)

class ImageInfoModel(object):
    """Model for image information."""

    def __init__(self, controller):
        """Initialize the ImageInfoModel.

        Args:
            controller: Reference to the controller
        """
        self.controller = controller
        self.processPipe = None

    def getProcessPipe(self):
        """Get the process pipe.

        Returns:
            processing.ProcessPipe: Current process pipe
        """
        return self.processPipe

    def setProcessPipe(self, processPipe):
        """Set the process pipe.

        Args:
            processPipe (processing.ProcessPipe): Process pipe to set
        """
        self.processPipe = processPipe

    def changeMeta(self, tagGroup, tag, on_off):
        """Change the metadata.

        Args:
            tagGroup: Group of the tag
            tag: Tag to change
            on_off (bool): New value of the tag
        """
        if pref.verbose: print(" [MODEL] >> ImageInfoModel.changeMeta(", tagGroup, ",", tag, ",", on_off, ")")
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
            self.processPipe.updateUserMeta(tagRootName, updatedMeta)
            if pref.verbose:
                print(" [MODEL] >> ImageInfoModel.changeUseCase()")
                for tt in updatedMeta: print(tt)

class CurveControlModel(object):
    """Model for curve control."""
    pass

class HDRviewerModel(object):
    """Model for HDR viewer."""

    def __init__(self, _controller):
        """Initialize the HDRviewerModel.

        Args:
            _controller: Reference to the controller
        """
        if pref.verbose: print(" [MODEL] >> HDRviewerModel.__init__()")
        self.controller = _controller
        self.currentIMG = None
        display_name = getattr(pref, "HDRdisplay", "none")
        display_config = getattr(pref, "HDRdisplays", {})
        self.displayModel = display_config[display_name]

    def scaling(self):
        """Get the scaling factor.

        Returns:
            int: Scaling factor
        """
        if pref.verbose: print(f" [MODEL] >> HDRviewerModel.scaling():{self.displayModel['scaling']}")
        return self.displayModel['scaling']

    def shape(self):
        """Get the shape of the display.

        Returns:
            tuple: Shape of the display
        """
        if pref.verbose: print(f" [MODEL] >> HDRviewerModel.shape():{self.displayModel['shape']}")
        return self.displayModel['shape']

class LchColorSelectorModel(object):
    """Model for LCH color selector."""

    def __init__(self, _controller):
        """Initialize the LchColorSelectorModel.

        Args:
            _controller: Reference to the controller
        """
        self.controller = _controller
        self.lightnessSelection = (0, 100) # min, max
        self.chromaSelection = (0, 100) # min, max
        self.hueSelection = (0, 360) # min, max
        self.exposure = 0.0
        self.hueShift = 0.0
        self.contrast = 0.0
        self.saturation = 0.0
        self.mask = False
        self.default = {
            "selection": {"lightness": [0, 100], "chroma": [0, 100], "hue": [0, 360]},
            "edit": {"hue": 0, "exposure": 0, "contrast": 0, "saturation": 0},
            "mask": False
        }

    def setHueSelection(self, hMin, hMax):
        """Set the hue selection range.

        Args:
            hMin: Minimum hue value
            hMax: Maximum hue value

        Returns:
            dict: Current values
        """
        self.hueSelection = hMin, hMax
        return self.getValues()

    def setChromaSelection(self, cMin, cMax):
        """Set the chroma selection range.

        Args:
            cMin: Minimum chroma value
            cMax: Maximum chroma value

        Returns:
            dict: Current values
        """
        self.chromaSelection = cMin, cMax
        return self.getValues()

    def setLightnessSelection(self, lMin, lMax):
        """Set the lightness selection range.

        Args:
            lMin: Minimum lightness value
            lMax: Maximum lightness value

        Returns:
            dict: Current values
        """
        self.lightnessSelection = lMin, lMax
        return self.getValues()

    def setExposure(self, ev):
        """Set the exposure value.

        Args:
            ev: Exposure value

        Returns:
            dict: Current values
        """
        self.exposure = ev
        return self.getValues()

    def setHueShift(self, hs):
        """Set the hue shift value.

        Args:
            hs: Hue shift value

        Returns:
            dict: Current values
        """
        self.hueShift = hs
        return self.getValues()

    def setContrast(self, contrast):
        """Set the contrast value.

        Args:
            contrast: Contrast value

        Returns:
            dict: Current values
        """
        self.contrast = contrast
        return self.getValues()

    def setSaturation(self, saturation):
        """Set the saturation value.

        Args:
            saturation: Saturation value

        Returns:
            dict: Current values
        """
        self.saturation = saturation
        return self.getValues()

    def setMask(self, value):
        """Set the mask value.

        Args:
            value: Mask value

        Returns:
            dict: Current values
        """
        self.mask = value
        return self.getValues()

    def getValues(self):
        """Get the current values.

        Returns:
            dict: Current values
        """
        return {
            'selection': {'lightness': self.lightnessSelection, 'chroma': self.chromaSelection, 'hue': self.hueSelection},
            'edit': {'hue': self.hueShift, 'exposure': self.exposure, 'contrast': self.contrast, 'saturation': self.saturation},
            'mask': self.mask
        }

    def setValues(self, values):
        """Set the values.

        Args:
            values (dict): Dictionary of values
        """
        self.lightnessSelection = values['selection']['lightness'] if 'lightness' in values['selection'].keys() else (0, 100)
        self.chromaSelection = values['selection']['chroma'] if 'chroma' in values['selection'].keys() else (0, 100)
        self.hueSelection = values['selection']['hue'] if 'hue' in values['selection'].keys() else (0, 360)
        self.exposure = values['edit']['exposure'] if 'exposure' in values['edit'].keys() else 0
        self.hueShift = values['edit']['hue'] if 'hue' in values['edit'].keys() else 0
        self.contrast = values['edit']['contrast'] if 'contrast' in values['edit'].keys() else 0
        self.saturation = values['edit']['saturation'] if 'saturation' in values['edit'].keys() else 0
        self.mask = values['mask'] if 'mask' in values.keys() else False

class GeometryModel(object):
    """Model for geometry adjustments."""

    def __init__(self, _controller):
        """Initialize the GeometryModel.

        Args:
            _controller: Reference to the controller
        """
        self.controller = _controller
        self.ratio = (16, 9)
        self.up = 0.0
        self.rotation = 0.0

    def setCroppingVerticalAdjustement(self, up):
        """Set the vertical cropping adjustment.

        Args:
            up: Vertical adjustment value

        Returns:
            dict: Current values
        """
        self.up = up
        return self.getValues()

    def setRotation(self, rotation):
        """Set the rotation value.

        Args:
            rotation: Rotation value

        Returns:
            dict: Current values
        """
        self.rotation = rotation
        return self.getValues()

    def getValues(self):
        """Get the current values.

        Returns:
            dict: Current values
        """
        return {'ratio': self.ratio, 'up': self.up, 'rotation': self.rotation}

    def setValues(self, values):
        """Set the values.

        Args:
            values (dict): Dictionary of values
        """
        self.ratio = values['ratio'] if 'ratio' in values.keys() else (16, 9)
        self.up = values['up'] if 'up' in values.keys() else 0.0
        self.rotation = values['rotation'] if 'rotation' in values.keys() else 0.0

# ------------------------------------------------------------------------------------------
# ---- Class AestheticsImageModel ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageAestheticsModel:
    """Class ImageAesthetics: encapsulates color palette (and related parameters), convexHull composition (and related parameters), etc.

    Attributes:
        parent (guiQt.controller.ImageAestheticsController): Controller
        processPipe (hdrCore.processing.ProcessPipe): Current selected process-pipe
        requireUpdate (bool): Flag to indicate if an update is required
        colorPalette (hdrCore.aesthetics.Palette): Color palette of the image

    Methods:
        __init__(parent): Initialize the ImageAestheticsModel
        getProcessPipe(): Get the current process pipe
        setProcessPipe(processPipe): Set the current process pipe
        endComputing(): End the computing process
        getPaletteImage(): Get the palette image
    """
    def __init__(self, parent):
        """Initialize the ImageAestheticsModel.

        Args:
            parent (guiQt.controller.ImageAestheticsController): Controller
        """
        if pref.verbose: print(" [MODEL] >> ImageAestheticsModel.__init__()")

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

    def getProcessPipe(self):
        """Get the current process pipe.

        Returns:
            hdrCore.processing.ProcessPipe: Current process pipe
        """
        return self.processPipe

    def setProcessPipe(self, processPipe):
        """Set the current process pipe.

        Args:
            processPipe (hdrCore.processing.ProcessPipe): Process pipe to set
        """
        if pref.verbose: print(" [MODEL] >> ImageAestheticsModel.setProcessPipe()")

        if processPipe != self.processPipe:
            self.processPipe = processPipe
            self.requireUpdate = True

        if self.requireUpdate:
            self.colorPalette = hdrCore.aesthetics.Palette.build(self.processPipe)
            # COMPUTE IMAGE OF PALETTE
            paletteIMG = self.colorPalette.createImageOfPalette()
            self.endComputing()
        else:
            pass

    def endComputing(self):
        """End the computing process."""
        self.requireUpdate = False

    def getPaletteImage(self):
        """Get the palette image.

        Returns:
            numpy.ndarray: Image of the color palette
        """
        return self.colorPalette.createImageOfPalette()

class ColorEditorsAutoModel:
    """Model for automatic color editors.

    Attributes:
        controller: Reference to the controller
        processStepId (str): Name of the process step
        nbColors (int): Number of colors to extract
        removeBlack (bool): Whether to remove black from the color extraction

    Methods:
        __init__(_controller, processStepName, nbColors, removeBlack): Initialize the ColorEditorsAutoModel
        compute(): Compute the color editors
    """
    def __init__(self, _controller, processStepName, nbColors, removeBlack=True):
        """Initialize the ColorEditorsAutoModel.

        Args:
            _controller: Reference to the controller
            processStepName (str): Name of the process step
            nbColors (int): Number of colors to extract
            removeBlack (bool): Whether to remove black from the color extraction
        """
        self.controller = _controller
        self.processStepId = processStepName
        self.nbColors = nbColors
        self.removeBlack = removeBlack

    def compute(self):
        """Compute the color editors.

        Returns:
            list[dict]: List of dictionaries containing color editor values
        """
        # get image according to processId
        processPipe = self.controller.parent.controller.getProcessPipe()
        if processPipe is not None:
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
            kmeans_cluster_Lab = sklearn.cluster.KMeans(n_clusters=self.nbColors + 1)
            kmeans_cluster_Lab.fit(LabPixelsVector)
            cluster_centers_Lab = kmeans_cluster_Lab.cluster_centers_
            cluster_labels_Lab = kmeans_cluster_Lab.labels_

            # remove darkness one
            idxLmin = np.argmin(cluster_centers_Lab[:, 0])  # idx of darkness
            cluster_centers_Lab = np.delete(cluster_centers_Lab, idxLmin, axis=0)  # remove min from cluster_centers_Lab

            # go to Lch
            cluster_centers_Lch = colour.Lab_to_LCHab(cluster_centers_Lab)

            # sort cluster by hue
            cluster_centersIdx = np.argsort(cluster_centers_Lch[:, 2])

            dictValuesList = []
            for j in range(len(cluster_centersIdx)):
                i = cluster_centersIdx[j]
                if j == 0:
                    Hmin = 0
                else:
                    Hmin = 0.5 * (cluster_centers_Lch[cluster_centersIdx[j - 1]][2] + cluster_centers_Lch[cluster_centersIdx[j]][2])
                if j == len(cluster_centersIdx) - 1:
                    Hmax = 360
                else:
                    Hmax = 0.5 * (cluster_centers_Lch[cluster_centersIdx[j]][2] + cluster_centers_Lch[cluster_centersIdx[j + 1]][2])

                Cmin = max(0, cluster_centers_Lch[cluster_centersIdx[j]][1] - 25)
                Cmax = min(100, cluster_centers_Lch[cluster_centersIdx[j]][1] + 25)

                Lmin = max(0, cluster_centers_Lch[cluster_centersIdx[j]][0] - 25)
                Lmax = min(100, cluster_centers_Lch[cluster_centersIdx[j]][0] + 25)

                dictSegment = {
                    "selection": {
                        "lightness": [Lmin, Lmax],
                        "chroma": [Cmin, Cmax],
                        "hue": [Hmin, Hmax]},
                    "edit": {"hue": 0, "exposure": 0, "contrast": 0, "saturation": 0},
                    "mask": False}
                dictValuesList.append(dictSegment)

            return dictValuesList

# ------------------------------------------------------------------------------------------






