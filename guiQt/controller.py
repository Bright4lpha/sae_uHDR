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

import enum, sys, subprocess, copy, colour, time, os, shutil, datetime, ctypes
import numpy as np
# pyQT5 import
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtWidgets import QMessageBox

from . import model, view, thread
import hdrCore.image, hdrCore.processing, hdrCore.utils
import hdrCore.coreC
import preferences.preferences as pref

# zj add for semi-auto curve 
import torch
from hdrCore.net import Net
from torch.autograd import Variable

# -----------------------------------------------------------------------------
# --- package methods ---------------------------------------------------------
# -----------------------------------------------------------------------------
def getScreenSize(app):
    screens = app.screens()
    res = list(map(lambda x: x.size(), screens))
    return res
# -----------------------------------------------------------------------------
# --- Class GalleryMode -------------------------------------------------------
# -----------------------------------------------------------------------------
class GalleryMode(enum.Enum):
    """ Enum  """

    _1x1         = 0         # 
    _3x2         = 1         # 
    _6x4         = 2         # 
    _9x6         = 3         #
    
    _2x1        =  4

    def nbRow(m):
        if m == GalleryMode._1x1: return 1
        if m == GalleryMode._3x2: return 2
        if m == GalleryMode._6x4: return 4
        if m == GalleryMode._9x6: return 6

        if m == GalleryMode._2x1: return 1
    def nbCol(m):
        if m == GalleryMode._1x1: return 1
        if m == GalleryMode._3x2: return 3
        if m == GalleryMode._6x4: return 6
        if m == GalleryMode._9x6: return 9

        if m == GalleryMode._2x1: return 2
# -----------------------------------------------------------------------------
# --- Class ImageWidgetController ---------------------------------------------
# -----------------------------------------------------------------------------
class ImageWidgetController:
    """ image widget controller """

    def __init__(self, image=None,id = -1):

        self.model = model.ImageWidgetModel(self)
        self.view = view.ImageWidgetView(self)

        self._id = id # store an (unique) id 

        if isinstance(image, (np.ndarray, hdrCore.image.Image)):
            self.model.setImage(image)
            self.view.setPixmap(self.model.getColorData())

    def setImage(self, image):
        self.model.setImage(image)
        return self.view.setPixmap(self.model.getColorData())

    def setQPixmap(self, qPixmap):
        self.view.setQPixmap(qPixmap)

    def id(self): return self._id
# -----------------------------------------------------------------------------
# --- Class ImageGalleryController --------------------------------------------
# -----------------------------------------------------------------------------
class ImageGalleryController():
    """ image gallery controller """

    def __init__(self, parent):
        if pref.verbose: print(" [CONTROL] >> ImageGalleryController.__init__()")

        self.parent = parent    # AppView
        self.view = view.ImageGalleryView(self)
        self.model = model.ImageGalleryModel(self)

    def setImages(self, imageFiles): self.model.setImages(imageFiles)

    def updateImages(self):
        """ called by ImageGalleryModel to ask for update images """
        self.view.pageNumber = 0 # reset page number
        self.view.updateImages()

    def callBackButton_previousPage(self):  
        self.view.changePageNumber(-1)
        if self.view.shapeMode == GalleryMode._1x1 : self.selectImage(0)

    def callBackButton_nextPage(self):      
        self.view.changePageNumber(+1)
        if self.view.shapeMode == GalleryMode._1x1 : self.selectImage(0)

    def computePageNumberOnGalleryModeChange(self,newGalleryMode):
        currentPage = self.view.pageNumber
        nbImagePerPage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        selectedImage = self.model.selectedImage() if (self.model.selectedImage()!=-1) else currentPage*nbImagePerPage
        newNbImagePerPage = GalleryMode.nbRow(newGalleryMode)*GalleryMode.nbCol(newGalleryMode)

        newPageNumber = selectedImage//newNbImagePerPage

        return newPageNumber

    def callBackButton_1x1(self):
        if self.view.shapeMode != GalleryMode._1x1:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._1x1)
            self.view.shapeMode = GalleryMode._1x1
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_3x2(self): 
        if self.view.shapeMode != GalleryMode._3x2:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._3x2)
            self.view.shapeMode = GalleryMode._3x2
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_6x4(self): 
        if self.view.shapeMode != GalleryMode._6x4:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._6x4)
            self.view.shapeMode = GalleryMode._6x4
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_9x6(self):
        if self.view.shapeMode != GalleryMode._9x6:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._9x6)
            self.view.shapeMode = GalleryMode._9x6
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def callBackButton_2x1(self):
        if self.view.shapeMode != GalleryMode._2x1:
            self.view.resetGridLayoutWidgets()
            self.view.pageNumber = self.computePageNumberOnGalleryModeChange(GalleryMode._2x1)
            self.view.shapeMode = GalleryMode._2x1
            self.view.buildGridLayoutWidgets()
            self.view.updateImages()
            self.model.loadPage(self.view.pageNumber)
            self.view.repaint()

    def selectImage(self, id):
        if pref.verbose: print(" [CONTROL] >> ImageGalleryController.selectImage()")

        nbImagePage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        idxImage = self.view.pageNumber*nbImagePage+id
        # check id
        if (idxImage < len(self.model.processPipes)):
            # update selected image
            processPipe = self.model.processPipes[idxImage]
            if processPipe:
                if self.parent.dock.setProcessPipe(processPipe):
                    self.model.setSelectedImage(idxImage)

    def getSelectedProcessPipe(self):
        if pref.verbose:  print(" [CONTROL] >> ImageGalleryController.getSelectedProcessPipe()")
        return self.model.getSelectedProcessPipe()

    def setProcessPipeWidgetQPixmap(self, qPixmap):
        idxProcessPipe = self.model.selectedImage()
        nbImagePage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        idxImageWidget = idxProcessPipe%nbImagePage
        if pref.verbose:  print(" >> ImageGalleryController.setProcessPipeWidgetQPixmap(...)[ image id:",idxProcessPipe,">> image widget controller:",idxImageWidget,"]")
        self.view.imagesControllers[idxImageWidget].setQPixmap(qPixmap)

    def save(self):
        if pref.verbose: print(" [CONTROL] >> ImageGalleryController.save()")
        self.model.save()

    def currentPage(self): return self.view.currentPage()

    def pageIdx(self):
        nb = self.currentPage()
        nbImagePage = GalleryMode.nbRow(self.view.shapeMode)*GalleryMode.nbCol(self.view.shapeMode)
        return (nb*nbImagePage), ((nb+1)*nbImagePage)

    def getFilenamesOfCurrentPage(self): return self.model.getFilenamesOfCurrentPage()

    def getProcessPipeById(self,i) : return self.model.getProcessPipeById(i)

    def getProcessPipes(self): return self.model.processPipes
# -----------------------------------------------------------------------------
# --- Class AppController -----------------------------------------------------
# -----------------------------------------------------------------------------
class AppController(object):
    """controller for MainWindow
    
        Attributes:
            screenSize 
            hdrDisplay 
            view                          
            model
            
        Methods:
            callBackSelectDir(self)
            callBackSave(self)
            callBackDisplayHDR(self)
            callBackEndDisplay(self, img)
            callBackCloseDisplayHDR(self)
            callBackCompareRawEditedHDR(self)
            callBackExportHDR(self)
            callBackEndExportHDR(self, img)
            callBackExportAllHDR(self)            
            callBackEndAllExportHDR(self, img)  

    """

    def __init__(self, app):
        if pref.verbose: print(" [CONTROL] >> AppController.__init__()")

        self.screenSize = getScreenSize(app)# get screens size

        # attributes
        self.hdrDisplay = HDRviewerController(self)
        self.view =  view.AppView(self, HDRcontroller = self.hdrDisplay)                         
        self.model = model.AppModel(self)

        self.dirName = None
        self.imagesName = []
        
        self.view.show()
    # -----------------------------------------------------------------------------

    def callBackSelectDir(self):
        """Callback of export HDR menu: open file dialog, store image filenames (self.imagesName), set directory to model
        """
        if pref.verbose: print(" [CONTROL] >> AppController.callBackSelectDir()")
        dirName = QFileDialog.getExistingDirectory(None, 'Select Directory', self.model.directory)
        if dirName != "":
            # save current images (metadata)
            self.view.imageGalleryController.save()
            # get images in the selected directory
            self.imagesName = []; self.imagesName = list(self.model.setDirectory(dirName))

            self.view.imageGalleryController.setImages(self.imagesName)
            self.hdrDisplay.displaySplash()
    # -----------------------------------------------------------------------------
    def callBackSave(self): self.view.imageGalleryController.save()
    # -----------------------------------------------------------------------------
    def callBackQuit(self):
        if pref.verbose: print(" [CB] >> AppController.callBackQuit()")
        self.view.imageGalleryController.save()
        self.hdrDisplay.close()
        sys.exit()
    # -----------------------------------------------------------------------------
    def callBackDisplayHDR(self):

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackDisplayHDR()")

        selectedProcessPipe = self.view.imageGalleryController.model.getSelectedProcessPipe()

        if selectedProcessPipe:
            self.view.statusBar().showMessage('displaying HDR image, full size image computation: start, please wait !')
            self.view.statusBar().repaint()
            # save current processpipe metada
            originalImage = copy.deepcopy(selectedProcessPipe.originalImage)
            originalImage.metadata.metadata['processpipe'] = selectedProcessPipe.toDict()
            originalImage.metadata.save()

            # load full size image
            img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = False 
            # make a copy of selectedProcessPipe  
            processpipe = copy.deepcopy(selectedProcessPipe)

            # set size to display size
            size = pref.getDisplayShape()
            img = img.process(hdrCore.processing.resize(),size=(None, size[1]))

            # set image to process-pipe
            processpipe.setImage(img)

            thread.cCompute(self.callBackEndDisplay, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)
    # -----------------------------------------------------------------------------
    def callBackEndDisplay(self, img):

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackEndDisplay()")

        # turn off: autoResize
        hdrCore.processing.ProcessPipe.autoResize = True  
        
        self.view.statusBar().showMessage('displaying HDR image, full size image computation: done !')

        # clip, scale
        img = img.process(hdrCore.processing.clip())
        img.colorData = img.colorData*pref.getDisplayScaling()

        colour.write_image(img.colorData,"temp.hdr", method='Imageio') # local copy for display
        self.hdrDisplay.displayFile("temp.hdr")
    # -----------------------------------------------------------------------------
    def callBackCloseDisplayHDR(self):
        if pref.verbose: print(" [CONTROL] >> AppController.callBackCloseDisplayHDR()")
        self.hdrDisplay.displaySplash()
    # -----------------------------------------------------------------------------
    def callBackCompareRawEditedHDR(self):
        """
        Callback of compare raw/edited HDR menu

        Display side by side original image and edited version.        
        """

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackCompareOriginalInputHDR()")

        # process real size image
        # get selected process pipe
        selectedProcessPipe = self.view.imageGalleryController.model.getSelectedProcessPipe()

        if selectedProcessPipe:         # check if a process pipe is selected

            # read original image
            img = hdrCore.image.Image.read(selectedProcessPipe.originalImage.path+'/'+selectedProcessPipe.originalImage.name)

            # resize
            screenY, screenX = pref.getDisplayShape()

            imgY, imgX,_ = img.shape

            marginY = int((screenY - imgY/2)/2)
            marginX = int(marginY/4)
            imgXp = int((screenX - 3*marginX)/2)
            img = img.process(hdrCore.processing.resize(),size=(None,imgXp))
            imgY, imgX, _ = img.shape

            # original image after resize
            ori = copy.deepcopy(img)

            # build process pipe from selected one them compute
            pp = hdrCore.processing.ProcessPipe()
            hdrCore.processing.ProcessPipe.autoResize = False   # stop autoResize
            params= []
            for p in selectedProcessPipe.processNodes: 
                pp.append(copy.deepcopy(p.process),paramDict=None, name=copy.deepcopy(p.name))
                params.append({p.name:p.params})
            img.metadata.metadata['processpipe'] = params
            pp.setImage(img)

            res = hdrCore.coreC.coreCcompute(img, pp)
            res = res.process(hdrCore.processing.clip())
            
            imgYres, imgXres, _ = res.colorData.shape

            hdrCore.processing.ProcessPipe.autoResize = True    # return to autoResize

            # make comparison image
            oriColorData = ori.colorData*pref.getDisplayScaling()
            resColorData = res.colorData*pref.getDisplayScaling()
            display = np.ones((screenY,screenX,3))*0.2
            marginY = int((screenY - imgY)/2)
            marginYres = int((screenY - imgYres)/2)

            display[marginY:marginY+imgY, marginX:marginX+imgX,:] = oriColorData
            display[marginYres:marginYres+imgYres, 2*marginX+imgX:2*marginX+imgX+imgXres,:] = resColorData
            
            # save as compOrigFinal.hdr
            colour.write_image(display,'compOrigFinal.hdr', method='Imageio')
            self.hdrDisplay.displayFile('compOrigFinal.hdr')
    # -----------------------------------------------------------------------------
    def callBackExportHDR(self):
        """
        Callback of export HDR menu

        Export the image associated to selected process pipe
        """

        if pref.verbose:  print(" [CONTROL] >> AppController.callBackExportHDR()") 

        selectedProcessPipe = self.view.imageGalleryController.model.getSelectedProcessPipe()


        if selectedProcessPipe:
            # select dir where to save export
            self.dirName = QFileDialog.getExistingDirectory(None, 'Select Directory where to export HDR file', self.model.directory)

            # show export message
            self.view.statusBar().showMessage('exporting HDR image ('+pref.getHDRdisplay()['tag']+'), full size image computation: start, please wait !')
            self.view.statusBar().repaint()

            # save current processpipe metada
            originalImage = copy.deepcopy(selectedProcessPipe.originalImage)
            originalImage.metadata.metadata['processpipe'] = selectedProcessPipe.toDict()
            originalImage.metadata.save()

            # load full size image
            img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = False 
            # make a copy of selectedProcessPipe  
            processpipe = copy.deepcopy(selectedProcessPipe)

            # set image to process-pipe
            processpipe.setImage(img)

            thread.cCompute(self.callBackEndExportHDR, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)
    # -----------------------------------------------------------------------------
    def callBackEndExportHDR(self, img):
        # turn off: autoResize
        hdrCore.processing.ProcessPipe.autoResize = True  
        
        self.view.statusBar().showMessage('exporting HDR image ('+pref.getHDRdisplay()['tag']+'), full size image computation: done !')

        # clip, scale
        img = img.process(hdrCore.processing.clip())
        img.colorData = img.colorData*pref.getDisplayScaling()

        if self.dirName:
            pathExport = os.path.join(self.dirName, img.name[:-4]+pref.getHDRdisplay()['post']+'.hdr')
            img.type = hdrCore.image.imageType.HDR
            img.metadata.metadata['processpipe'] = None
            img.metadata.metadata['display'] = pref.getHDRdisplay()['tag']

            img.write(pathExport)

        colour.write_image(img.colorData,"temp.hdr", method='Imageio') # local copy for display
        self.hdrDisplay.displayFile("temp.hdr")
    # -----------------------------------------------------------------------------
    def callBackExportAllHDR(self):
        if pref.verbose:  print(" [CONTROL] >> AppController.callBackExportAllHDR()")

        self.processPipes = self.view.imageGalleryController.getProcessPipes()

        # select dir where to save export
        self.dirName = QFileDialog.getExistingDirectory(None, 'Select Directory where to export HDR file', self.model.directory)
        self.view.statusBar().showMessage('exporting '+str(len(self.processPipes))+' HDR images ... please wait')
        self.view.statusBar().repaint()
        self.imageToExport = len(self.processPipes) ; self.imageExportDone = 0

        pp = self.processPipes[0]

        # save current processpipe metada
        originalImage = copy.deepcopy(pp.originalImage)
        originalImage.metadata.metadata['processpipe'] = pp.toDict()
        originalImage.metadata.save()

        # load full size image
        img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

        # turn off: autoResize
        hdrCore.processing.ProcessPipe.autoResize = False 
        # make a copy of selectedProcessPipe  
        processpipe = copy.deepcopy(pp)

        # set image to process-pipe
        processpipe.setImage(img)

        thread.cCompute(self.callBackEndAllExportHDR, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)            
    # -----------------------------------------------------------------------------
    def callBackEndAllExportHDR(self, img):
        # last image ?
        self.imageExportDone +=1

        self.view.statusBar().showMessage('exporting HDR images ('+pref.getHDRdisplay()['tag']+'):'+str(int(100*self.imageExportDone/self.imageToExport))+'% done !')

        # clip, scale
        img = img.process(hdrCore.processing.clip())
        img.colorData = img.colorData*pref.getDisplayScaling()

        if self.dirName:
            pathExport = os.path.join(self.dirName, img.name[:-4]+pref.getHDRdisplay()['post']+'.hdr')
            img.type = hdrCore.image.imageType.HDR
            img.metadata.metadata['processpipe'] = None
            img.metadata.metadata['display'] = pref.getHDRdisplay()['tag']

            img.write(pathExport)

        if self.imageExportDone == self.imageToExport :
            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = True
        else:
            pp = self.processPipes[self.imageExportDone]

            if not pp:
                img = hdrCore.image.Image.read(self.imagesName[self.imageExportDone], thumb=True)
                pp = model.EditImageModel.buildProcessPipe()
                pp.setImage(img)                      


            # save current processpipe metada
            originalImage = copy.deepcopy(pp.originalImage)
            originalImage.metadata.metadata['processpipe'] = pp.toDict()
            originalImage.metadata.save()

            # load full size image
            img = hdrCore.image.Image.read(originalImage.path+'/'+originalImage.name)

            # turn off: autoResize
            hdrCore.processing.ProcessPipe.autoResize = False 
            # make a copy of selectedProcessPipe  
            processpipe = copy.deepcopy(pp)

            # set image to process-pipe
            processpipe.setImage(img)

            thread.cCompute(self.callBackEndAllExportHDR, processpipe, toneMap=False, progress=self.view.statusBar().showMessage)            
# ------------------------------------------------------------------------------------------
# --- class MultiDockController() ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class MultiDockController():
    def __init__(self,parent=None, HDRcontroller = None):
        if pref.verbose: print(" [CONTROL] >> MultiDockController.__init__()")

        self.parent = parent
        self.view = view.MultiDockView(self, HDRcontroller)
        self.model = None
    # ---------------------------------------------------------------------------------------
    def activateEDIT(self): self.switch(0)
    def activateINFO(self): self.switch(1)
    def activateMIAM(self):  self.switch(2)
    # ---------------------------------------------------------------------------------------
    def switch(self,nb):
        if pref.verbose:  print(" [CONTROL] >> MultiDockController.switch()")
        self.view.switch(nb)
    # --------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        if pref.verbose: print(" [CONTROL] >> MultiDockController.setProcessPipe(",processPipe.getImage().name,")")

        return self.view.setProcessPipe(processPipe)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class EditImageController:

    def __init__(self, parent=None, HDRcontroller = None):
        if pref.verbose: print(" [CONTROL] >> EditImageController.__init__(",")")

        self.parent = parent

        self.previewHDR = True
        self.controllerHDR = HDRcontroller

        self.view = view.EditImageView(self)
        self.model = model.EditImageModel(self)
    # -----------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        if pref.verbose: print(" [CONTROL] >> EditImageController.setProcessPipe(",")")

        if self.model.setProcessPipe(processPipe):

            # update view
            self.view.setProcessPipe(processPipe)
            self.view.imageWidgetController.setImage(processPipe.getImage())

            self.view.plotToneCurve()

            # update hdr viewer
            self.controllerHDR.displaySplash()

            return True
        else:
            return False
    # -----------------------------------------------------------------------------
    def getProcessPipe(self) : return self.model.getProcessPipe()
    # -----------------------------------------------------------------------------
    def buildView(self,processPipe=None):
        if pref.verbose: print(" [CONTROL] >> EditImageController.buildView(",")")

        """ called when MultiDockController recall a controller/view """
        self.view = view.EditImageView(self, build=True)
        if processPipe: self.setProcessPipe(processPipe)
    # -----------------------------------------------------------------------------
    def autoExposure(self): 
        if pref.verbose: print(" [CONTROL] >> EditImageController.autoExposure(",")")
        if self.model.processpipe:
      
            img = self.model.autoExposure()

            # update view with computed EV
            paramDict = self.model.getEV()
            self.view.exposure.setValue(paramDict['EV'])

            qPixmap =  self.view.setImage(img)
            self.parent.controller.parent.controller.view.imageGalleryController.setProcessPipeWidgetQPixmap(qPixmap)
    # -----------------------------------------------------------------------------
    def changeExposure(self,value):
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeExposure(",value,")")
        if self.model.processpipe: self.model.changeExposure(value)
    # -----------------------------------------------------------------------------
    def changeContrast(self,value):
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeContrast(",value,")")
        if self.model.processpipe: self.model.changeContrast(value)
    # -----------------------------------------------------------------------------
    def changeToneCurve(self,controlPoints):
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeToneCurve("")")
        if self.model.processpipe: self.model.changeToneCurve(controlPoints)
    # -----------------------------------------------------------------------------
    def changeLightnessMask(self, maskValues):
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeLightnessMask(",maskValues,")")
        if self.model.processpipe: self.model.changeLightnessMask(maskValues)
    # -----------------------------------------------------------------------------
    def changeSaturation(self,value):
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeSaturation(",value,")")
        if self.model.processpipe: self.model.changeSaturation(value)
    # -----------------------------------------------------------------------------
    def changeColorEditor(self,values, idName):
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeColorEditor(",values,")")
        if self.model.processpipe: self.model.changeColorEditor(values, idName)
    # -----------------------------------------------------------------------------
    def changeGeometry(self,values):
        if pref.verbose: print(" [CONTROL] >> EditImageController.changeGeometry(",values,")")
        if self.model.processpipe: self.model.changeGeometry(values)
    # -----------------------------------------------------------------------------
    def updateImage(self,imgTM):
        """
        updateImage: called when process-pipe computation is done
            
        Args:
            imgTM (hdrCoreimage.Image, required): tone mapped image (resized) for GUI display
        """
        qPixmap =  self.view.setImage(imgTM)
        self.parent.controller.parent.controller.view.imageGalleryController.setProcessPipeWidgetQPixmap(qPixmap)
        self.view.plotToneCurve()

        # if aesthetics model > notify required update


        if self.previewHDR and self.model.autoPreviewHDR:
            self.controllerHDR.callBackUpdate()
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageInfoController:

    def __init__(self, parent=None):
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.__init__()")

        self.parent = parent
        self.view = view.ImageInfoView(self)
        self.model = model.ImageInfoModel(self)

        self.callBackActive = True
    # -----------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.setProcessPipe(",processPipe.getImage().name,")")
        self.model.setProcessPipe(processPipe)
        self.view.setProcessPipe(processPipe)
        return True
    # -----------------------------------------------------------------------------
    def buildView(self,processPipe=None):
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.buildView()")

        """ called when MultiDockController recall a controller/view """
        self.view = view.ImageInfoView(self)
        if processPipe: self.setProcessPipe(processPipe)
    # -----------------------------------------------------------------------------
    def metadataChange(self,metaGroup,metaTag, on_off): 
        if pref.verbose: print(" [CONTROL] >> ImageInfoController.useCaseChange(",metaGroup,",", metaTag,",", on_off,")")
        self.model.changeMeta(metaGroup,metaTag, on_off)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceSliderController():
    def __init__(self, parent,name, defaultValue, range, step,callBackValueChange=None,callBackAutoPush= None):
        if pref.verbose: print(" [CONTROL] >> AdvanceSliderController.__init__(",") ")
        self.parent = parent

        self.view = view.AdvanceSliderView(self,  name, defaultValue, range, step)
        self.model = model.AdvanceSliderModel(self, value=defaultValue)

        self.step = step
        self.defaultValue = defaultValue
        self.range = range

        self.callBackActive = True
        self.callBackValueChange = callBackValueChange
        self.callBackAutoPush = callBackAutoPush
    # -----------------------------------------------------------------------------
    def sliderChange(self):

        value = self.view.slider.value()*self.step

        if pref.verbose: print(" [CB] >> AdvanceSliderController.sliderChange(",value,")[callBackActive:",self.callBackActive,"] ")

        self.model.value = value
        self.view.editValue.setText(str(value))
        if self.callBackActive and self.callBackValueChange: self.callBackValueChange(value)
    # -----------------------------------------------------------------------------
    def setValue(self, value, callBackActive = True):
        if pref.verbose: print(" [CONTROL] >> AdvanceSliderController.setValue(",value,") ")

        """ set value value in 'model' range"""
        self.callBackActive = callBackActive
        self.view.slider.setValue(int(value/self.step))
        self.view.editValue.setText(str(value))
        self.model.setValue(int(value))

        self.callBackActive = True
    # -----------------------------------------------------------------------------
    def reset(self):
        if pref.verbose : print(" [CB] >> AdvanceSliderController.reset(",") ")

        self.setValue(self.defaultValue,callBackActive = False)
        if self.callBackValueChange: self.callBackValueChange(self.defaultValue)
    # -----------------------------------------------------------------------------
    def auto(self):
        if pref.verbose: print(" [CB] >> AdvanceSliderController.auto(",") ")

        if self.callBackAutoPush: self.callBackAutoPush()
# ------------------------------------------------------------------------------------------
# --- class AdvanceSliderController --------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ToneCurveController():
    def __init__(self, parent):

        self.parent = parent
        self.model = model.ToneCurveModel()
        self.view = view.ToneCurveView(self)
        self.callBackActive = True

        # tone curve display control
        self.showInput =False
        self.showbefore = False
        self.showAfter = False
        self.showOutput = True

        # zj add semi-auto curve
        # machine learning network and weight file 
        self.weightFile = 'MSESig505_0419.pth'
        self.networkModel = None  
    # -----------------------------------------------------------------------------
    def sliderChange(self, key, value):
        if pref.verbose: print(" [CB] >> ToneCurveController.sliderChange(",key,",",value,")[callBackActive:",self.callBackActive,"] ")

        if self.callBackActive:
            newValues = self.model.setValue(key, value, autoScale = False)
            self.parent.controller.changeToneCurve(newValues) 
  
            points = self.model.evaluate()

            self.callBackActive =  False

            self.view.sliderShadows.setValue(newValues["shadows"][1])
            self.view.editShadows.setText(str(newValues["shadows"][1]))

            self.view.sliderBlacks.setValue(newValues["blacks"][1])
            self.view.editBlacks.setText(str(newValues["blacks"][1]))

            self.view.sliderMediums.setValue(newValues["mediums"][1])
            self.view.editMediums.setText(str(newValues["mediums"][1]))

            self.view.sliderWhites.setValue(newValues["whites"][1])
            self.view.editWhites.setText(str(newValues["whites"][1]))

            self.view.sliderHighlights.setValue(newValues["highlights"][1])
            self.view.editHighlights.setText(str(newValues["highlights"][1]))

            self.callBackActive =  True
    # -----------------------------------------------------------------------------
    def setValues(self, valuesDict,callBackActive = False):
        if pref.verbose: print(" [CONTROL] >> ToneCurveController.setValue(",valuesDict,") ")

        self.callBackActive = callBackActive

        self.model.setValues(valuesDict)
        points = self.model.evaluate()

        self.view.sliderShadows.setValue(valuesDict["shadows"][1])
        self.view.editShadows.setText(str(valuesDict["shadows"][1]))

        self.view.sliderBlacks.setValue(valuesDict["blacks"][1])
        self.view.editBlacks.setText(str(valuesDict["blacks"][1]))

        self.view.sliderMediums.setValue(valuesDict["mediums"][1])
        self.view.editMediums.setText(str(valuesDict["mediums"][1]))

        self.view.sliderWhites.setValue(valuesDict["whites"][1])
        self.view.editWhites.setText(str(valuesDict["whites"][1]))

        self.view.sliderHighlights.setValue(valuesDict["highlights"][1])
        self.view.editHighlights.setText(str(valuesDict["highlights"][1]))

        self.callBackActive = True
    # -----------------------------------------------------------------------------     
    # zj add for semi-auto curve begin
    def autoCurve(self):
        processPipe = self.parent.controller.model.getProcessPipe()
        if processPipe != None :
            idExposure = processPipe.getProcessNodeByName("tonecurve")
            bins = np.linspace(0,1,50+1)

            imageBeforeColorData = processPipe.processNodes[idExposure-1].outputImage.colorData
            imageBeforeColorData[imageBeforeColorData>1]=1
            imageBeforeY = colour.sRGB_to_XYZ(imageBeforeColorData, apply_cctf_decoding=False)[:,:,1]
            nphistBefore  = np.histogram(imageBeforeY, bins)[0]
            nphistBefore  = nphistBefore/np.amax(nphistBefore)

            npImgHistCumuNorm = np.empty_like(nphistBefore)
            npImgHistCumu = np.cumsum(nphistBefore)
            npImgHistCumuNorm = npImgHistCumu/np.max(npImgHistCumu)
            
            #predict keypoint value
            if self.networkModel == None:
                self.networkModel = Net(50,5)
                self.networkModel.load_state_dict(torch.load(self.weightFile))
                self.networkModel.eval()

            with torch.no_grad():
                x = Variable(torch.FloatTensor([npImgHistCumuNorm.tolist(),]), requires_grad=True)
                y_predict = self.networkModel(x)

            kpc = (y_predict[0]*100).tolist() 
            kpcDict = {'start':[0.0,0.0], 'shadows': [10.0,kpc[0]], 'blacks': [30.0,kpc[1]], 'mediums': [50.0,kpc[2]], 'whites': [70.0,kpc[3]], 'highlights': [90.0,kpc[4]], 'end': [100.0,100.0]}
            self.setValues(kpcDict,callBackActive = True)
            self.parent.controller.changeToneCurve(kpcDict) 
     
    # zj add for semi-auto curve end   

    # -----------------------------------------------------------------------------
    def reset(self, key):
        if pref.verbose: print(" [CONTROL] >> ToneCurveController.reset(",key,") ")

        valuesDefault = copy.deepcopy(self.model.default[key])[1]
        controls = self.model.setValue(key, valuesDefault)
        self.setValues(controls,callBackActive = False)

        self.parent.controller.changeToneCurve(controls) 
    # -----------------------------------------------------------------------------
    def plotCurve(self):
        try:
            self.view.curve.plot([0,100],[0,100],'r--', clear=True)
            self.view.curve.plot([20,20],[0,100],'r--', clear=False)
            self.view.curve.plot([40,40],[0,100],'r--', clear=False)
            self.view.curve.plot([60,60],[0,100],'r--', clear=False)
            self.view.curve.plot([80,80],[0,100],'r--', clear=False)

            processPipe = self.parent.controller.model.getProcessPipe()
            idExposure = processPipe.getProcessNodeByName("tonecurve")

            bins = np.linspace(0,1,50+1)

            if self.showInput:
                imageInput = copy.deepcopy(processPipe.getInputImage())
                if imageInput.linear: imageInputColorData =colour.cctf_encoding(imageInput.colorData, function='sRGB')
                else: imageInputColorData = imageInput.colorData
                imageInputColorData[imageInputColorData>1]=1
                imageInputY = colour.sRGB_to_XYZ(imageInputColorData, apply_cctf_decoding=False)[:,:,1]
                nphistInput  = np.histogram(imageInputY, bins)[0]
                nphistInput  = nphistInput/np.amax(nphistInput)
                self.view.curve.plot(bins[:-1]*100,nphistInput*100,'k--',  clear=False)

            if self.showbefore:
                imageBeforeColorData = processPipe.processNodes[idExposure-1].outputImage.colorData
                imageBeforeColorData[imageBeforeColorData>1]=1
                imageBeforeY = colour.sRGB_to_XYZ(imageBeforeColorData, apply_cctf_decoding=False)[:,:,1]
                nphistBefore  = np.histogram(imageBeforeY, bins)[0]
                nphistBefore  = nphistBefore/np.amax(nphistBefore)
                self.view.curve.plot(bins[:-1]*100,nphistBefore*100,'b--',  clear=False)

            if self.showAfter:
                imageAftercolorData = processPipe.processNodes[idExposure].outputImage.colorData
                imageAftercolorData[imageAftercolorData>1]=1
                imageAfterY  = colour.sRGB_to_XYZ(imageAftercolorData,  apply_cctf_decoding=False)[:,:,1]
                nphistAfter   = np.histogram(imageAfterY, bins)[0]
                nphistAfter   =nphistAfter/np.amax(nphistAfter)
                self.view.curve.plot(bins[:-1]*100,nphistAfter*100,'b',     clear=False)

            if self.showOutput:
                imageAftercolorData = processPipe.getImage(toneMap=True).colorData
                imageAftercolorData[imageAftercolorData>1]=1
                imageAfterY  = colour.sRGB_to_XYZ(imageAftercolorData,  apply_cctf_decoding=False)[:,:,1]
                nphistAfter   = np.histogram(imageAfterY, bins)[0]
                nphistAfter   =nphistAfter/np.amax(nphistAfter)
                self.view.curve.plot(bins[:-1]*100,nphistAfter*100,'b',     clear=False)

            controlPointCoordinates= np.asarray(list(self.model.control.values()))
            self.view.curve.plot(controlPointCoordinates[1:-1,0],controlPointCoordinates[1:-1,1],'ro', clear=False)
            points = np.asarray(self.model.curve.evalpts)
            x = points[:,0]
            self.view.curve.plot(points[x<100,0],points[x<100,1],'r',clear=False)
        except:
            time.sleep(0.5)
            self.plotCurve()
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class LightnessMaskController():
    def __init__(self, parent):
        if pref.verbose: print(" [CONTROL] >> MaskLightnessController.__init__(",")")

        self.parent= parent
        self.model = model.LightnessMaskModel(self)
        self.view = view.LightnessMaskView(self)

        self.callBackActive = True
    # -----------------------------------------------------------------------------
    def maskChange(self,key, on_off):
        if pref.verbose: print(" [CB] >> MaskLightnessController.maskChange(",key,",",on_off,")[callBackActive:",self.callBackActive,"] ")

        maskState = self.model.maskChange(key, on_off)  
        self.parent.controller.changeLightnessMask(maskState) 
    # -----------------------------------------------------------------------------
    def setValues(self, values,callBackActive = False):
        if pref.verbose: print(" [CONTROL] >> LightnessMaskController.setValue(",values,") ")

        self.callBackActive = callBackActive

        self.model.setValues(values)

        self.view.checkboxShadows.setChecked(values["shadows"])
        self.view.checkboxBlacks.setChecked(values["blacks"])
        self.view.checkboxMediums.setChecked(values["mediums"])
        self.view.checkboxWhites.setChecked(values["whites"])
        self.view.checkboxHighlights.setChecked(values["highlights"])   

        self.callBackActive = True
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class HDRviewerController():
    def __init__(self, parent):
        if pref.verbose: print(" [CONTROL] >> HDRviewerController.__init__(",")")

        self.parent= parent
        self.model = model.HDRviewerModel(self)
        self.view = None 

        self.viewerProcess = None

        self.displaySplash()

    def setView(self, view): self.view = view

    def callBackUpdate(self):
        selectedProcessPipe = self.parent.view.imageGalleryController.model.getSelectedProcessPipe()
        img = selectedProcessPipe.getImage(toneMap = False)
        self.displayIMG(img)
        self.model.currentIMG = img

    def callBackAuto(self,on_off):
        self.parent.view.dock.view.childControllers[0].model.autoPreviewHDR = on_off

    def callBackCompare(self):
        if self.model.currentIMG:
            old = self.model.currentIMG
            old = old.process(hdrCore.processing.clip())

            sp = self.parent.view.imageGalleryController.model.getSelectedProcessPipe()
            img = sp.getImage(toneMap = False)
            img = img.process(hdrCore.processing.clip())


            h1, w1, _ = old.colorData.shape
            h2, w2, _ = img.colorData.shape
            hD, wD = self.model.displayModel['shape']
            hM = int((hD - max(h1,h2))/2)
            wM = int((wD - (w1+w2))/3)
            back =  np.ones((hD,wD,3))*0.2

            back[hM:hM+h1,wM:wM+w1,:] = old.colorData*self.model.displayModel['scaling']
            back[hM:hM+h2,2*wM+w1:2*wM+w1+w2,:] = img.colorData*self.model.displayModel['scaling']

            # save as temp.hdr
            colour.write_image(back,'temp.hdr', method='Imageio')
            self.displayFile('temp.hdr')

            self.model.currentIMG = img
        else: self.callBackUpdate()

    def displayFile(self, HDRfilename):
        """
        HDRviewerController display file: run HDRImageViewer process ti display HDR image (from filename)

        Args:
            HDRfilename: string
                Required  : hdr image filename
                
        """

         # check that no current display process already open
        if self.viewerProcess:
            # the display HDR process is already running
            # close current
            subprocess.run(['taskkill', '/F', '/T', '/IM', "HDRImageViewer*"],capture_output=False)
            time.sleep(0.05)
        # run display HDR process
        self.viewerProcess = subprocess.Popen(["HDRImageViewer.exe","-f", "-input:"+HDRfilename, "-f", "-h"], shell=True)
        time.sleep(0.10)
        psData = subprocess.run(['tasklist'], capture_output=True, universal_newlines=True).stdout
        if not 'HDRImageViewer' in psData: 
            # re-run display HDR process
            self.viewerProcess = subprocess.Popen(["HDRImageViewer.exe","-f", "-input:"+HDRfilename, "-f", "-h"], shell=True)

    def displayIMG(self, img):
        img = img.process(hdrCore.processing.clip())
        colorData = img.colorData*self.model.displayModel['scaling']

        h,w, _ = colorData.shape
        hD, wD = self.model.displayModel['shape']

        if w<wD:
            back = np.ones((hD,wD,3))*0.2
            marginW = int((wD-w)/2)
            marginH = int((hD-h)/2)

            back[marginH:marginH+h,marginW:marginW+w,:]=colorData

        # save as temp.hdr
        colour.write_image(back,'temp.hdr', method='Imageio')
        self.displayFile('temp.hdr')

    def displaySplash(self):
        self.model.currentIMG = None
        self.displayFile('grey.hdr')

    def close(self):
        # check that no current display process already open
        if self.viewerProcess:
            # the display HDR process is already running
            # close current
            subprocess.run(['taskkill', '/F', '/T', '/IM', "HDRImageViewer*"],capture_output=False)
            self.viewerProcess = None
# ------------------------------------------------------------------------------------------
# ---- Class LchColorSelectorController ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class LchColorSelectorController:
    def __init__(self, parent, idName = None):
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.__init__(",") ")
        self.parent = parent
        self.model =    model.LchColorSelectorModel(self)
        self.view =     view.LchColorSelectorView(self)

        self.idName = idName

        self.callBackActive = True

    def sliderHueChange(self, vMin, vMax):
        values  = self.model.setHueSelection(vMin,vMax)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderChromaChange(self, vMin, vMax):
        values  = self.model.setChromaSelection(vMin,vMax)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderLightnessChange(self, vMin, vMax):
        values  = self.model.setLightnessSelection(vMin,vMax)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderExposureChange(self, ev):
        values = self.model.setExposure(ev)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderSaturationChange(self, sat):
        values = self.model.setSaturation(sat)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderContrastChange(self, cc):
        values = self.model.setContrast(cc)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def sliderHueShiftChange(self, hs):
        values = self.model.setHueShift(hs)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def checkboxMaskChange(self,value): 
        values = self.model.setMask(value)
        if self.callBackActive : self.parent.controller.changeColorEditor(values, self.idName)

    def setValues(self, values, callBackActive = False):
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.setValue(",values,") ")

        self.callBackActive = callBackActive
        # slider hue selection
        v = values['selection']['hue'] if 'hue' in values['selection'].keys() else (0,360)
        self.view.sliderHueMin.setValue(int(v[0]))
        self.view.sliderHueMax.setValue(int(v[1]))

        # slider chroma selection
        v = values['selection']['chroma'] if 'chroma' in values['selection'].keys() else (0,100)
        self.view.sliderChromaMin.setValue(v[0])
        self.view.sliderChromaMax.setValue(v[1])

        # slider lightness
        v = values['selection']['lightness'] if 'lightness' in values['selection'].keys() else (0,100)
        self.view.sliderLightMin.setValue(int(v[0]*3))
        self.view.sliderLightMax.setValue(int(v[1]*3))

        # hue shift editor
        v = values['edit']['hue'] if 'hue' in values['edit'].keys() else 0
        self.view.sliderHueShift.setValue(int(v))
        self.view.valueHueShift.setText(str(v)) 

        # exposure editor
        v : int = values['edit']['exposure'] if 'exposure' in values['edit'].keys() else 0
        self.view.sliderExposure.setValue(int(v*30))
        self.view.valueExposure.setText(str(v)) 

        # contrast editor
        v : int = values['edit']['contrast'] if 'contrast' in values['edit'].keys() else 0
        self.view.sliderContrast.setValue(int(v))
        self.view.valueContrast.setText(str(v))  

        # saturation editor
        v : int = values['edit']['saturation'] if 'saturation' in values['edit'].keys() else 0
        self.view.sliderSaturation.setValue(int(v))
        self.view.valueSaturation.setText(str(v))  

        # mask
        v : bool = values['mask'] if 'mask' in values.keys() else False
        self.view.checkboxMask.setChecked(values['mask'])             

        self.model.setValues(values)

        self.callBackActive = True

    # -----
    def resetSelection(self): 
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.resetSelection(",") ")

        default = copy.deepcopy(self.model.default)
        current = copy.deepcopy(self.model.getValues())
        
        current['selection'] = default['selection']

        self.setValues(current,callBackActive = True)
        self.callBackActive = True

    def resetEdit(self): 
        if pref.verbose: print(" [CONTROL] >> LchColorSelectorController.resetEdit(",") ")

        default = copy.deepcopy(self.model.default)
        current = copy.deepcopy(self.model.getValues())
        
        current['edit'] = default['edit']

        self.setValues(current,callBackActive = True)
        self.callBackActive = True
# ------------------------------------------------------------------------------------------
# ---- Class LchColorSelectorController ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class GeometryController:
    def __init__(self, parent ):
        if pref.verbose: print(" [CONTROL] >> GeometryController.__init__(",") ")
        self.parent = parent
        self.model =    model.GeometryModel(self)
        self.view =     view.GeometryView(self)

        self.callBackActive = True
    # callbacks
    def sliderCroppingVerticalAdjustementChange(self,v):
        values = self.model.setCroppingVerticalAdjustement(v)
        if self.callBackActive : self.parent.controller.changeGeometry(values)

    def sliderRotationChange(self,v):
        values = self.model.setRotation(v)
        if self.callBackActive : self.parent.controller.changeGeometry(values)

    def setValues(self, values, callBackActive = False):
        if pref.verbose: print(" [CONTROL] >> GeometryController.setValue(",values,") ")

        up =        values['up']        if 'up' in values.keys()        else 0.0
        rotation =  values['rotation']  if 'rotation' in values.keys()  else 0.0

        self.callBackActive = callBackActive

        self.view.sliderCroppingVerticalAdjustement.setValue(int(up))
        self.view.sliderRotation.setValue(int(rotation*6))
       
        self.model.setValues(values)

        self.callBackActive = True
# ------------------------------------------------------------------------------------------
# ---- Class AestheticsImageController -----------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageAestheticsController:
    def __init__(self, parent=None, HDRcontroller = None):
        if pref.verbose: print(" [CONTROL] >> AestheticsImageController.__init__(",")")

        self.parent = parent
        self.model = model.ImageAestheticsModel(self)
        self.view = view.ImageAestheticsView(self)
    # --------------------------------------------------------------------------------------
    def buildView(self,processPipe=None):
        if pref.verbose: print(" [CONTROL] >> AestheticsImageController.buildView()")

        # called when MultiDockController recall a controller/view 
        self.view = view.ImageAestheticsView(self)
        if processPipe: self.setProcessPipe(processPipe)
    # --------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe): 
        if pref.verbose: print(" [CONTROL] >> AestheticsImageController.setProcessPipe()")

        self.model.setProcessPipe(processPipe)
        self.view.setProcessPipe(processPipe, self.model.getPaletteImage())

        return True
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# --- message widget functions -------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def messageBox(title, text):
        msg = QMessageBox()
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setEscapeButton(QMessageBox.Close)
        msg.exec_()
# -----------------------------------------------------------------------------
def okCancelBox(title, text):
        msg = QMessageBox()
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setEscapeButton(QMessageBox.Close)
        return msg.exec_()
# ------------------------------------------------------------------------------------------
# ---- Class ColorEditorsAutoController ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class  ColorEditorsAutoController:
    def __init__(self, parent, controlledColorEditors, stepName ):
        if pref.verbose: print(" [CONTROL] >> ColorEditorsAutoController.__init__(",") ")

        self.parent = parent
        self.controlled = controlledColorEditors
        self.stepName =stepName

        self.model =    model.ColorEditorsAutoModel(self, stepName,len(controlledColorEditors), removeBlack= True)
        self.view =     view.ColorEditorsAutoView(self)

        self.callBackActive = True
    # callbacks
    def auto(self): 
        if pref.verbose: print(" [CONTROL] >> ColorEditorsAutoController.auto(",") ")
        for ce in self.controlled: ce.resetSelection(); ce.resetEdit()
        values = self.model.compute()

        if values != None:
            for i,v in enumerate(values): self.controlled[i].setValues(v, callBackActive = False)

