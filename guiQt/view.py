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
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow, QSplitter, QFrame, QDockWidget, QDesktopWidget
from PyQt5.QtWidgets import QSplitter, QFrame, QSlider, QCheckBox, QGroupBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QLayout, QScrollArea, QFormLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit,QLineEdit, QComboBox, QSpinBox
from PyQt5.QtWidgets import QAction, QProgressBar, QDialog
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtWidgets 

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from datetime import datetime
import time

import numpy as np
import hdrCore.image, hdrCore.processing
import math, enum
import functools

from . import controller, model
import hdrCore.metadata
import preferences.preferences as pref

# ------------------------------------------------------------------------------------------
# --- class ImageWidgetView(QWidget) -------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageWidgetView(QWidget):
    """description of class"""

    def __init__(self,controller,colorData = None):
        super().__init__()
        self.controller = controller
        self.label = QLabel(self)   # create a QtLabel for pixmap
        if not isinstance(colorData, np.ndarray): colorData = ImageWidgetView.emptyImageColorData()
        # self.colorData = colorData  # image content attributes           
        self.setPixmap(colorData)  

    def resize(self):
        self.label.resize(self.size())
        self.label.setPixmap(self.imagePixmap.scaled(self.size(),Qt.KeepAspectRatio))

    def resizeEvent(self,event):
        self.resize()
        super().resizeEvent(event)

    def setPixmap(self,colorData):
        if not isinstance(colorData, np.ndarray): 
            colorData = ImageWidgetView.emptyImageColorData()
        # self.colorData = colorData
        height, width, channel = colorData.shape   # compute pixmap
        bytesPerLine = channel * width
        # clip
        colorData[colorData>1.0] = 1.0
        colorData[colorData<0.0] = 0.0

        qImg = QImage((colorData*255).astype(np.uint8), width, height, bytesPerLine, QImage.Format_RGB888) # QImage
        self.imagePixmap = QPixmap.fromImage(qImg)
        self.resize()

        return self.imagePixmap

    def setQPixmap(self, qPixmap):
        self.imagePixmap = qPixmap
        self.resize()

    def emptyImageColorData(): return np.ones((90,160,3))*(220/255) 
# ------------------------------------------------------------------------------------------
# --- class FigureWidget(FigureCanvas ------------------------------------------------------
# ------------------------------------------------------------------------------------------
class FigureWidget(FigureCanvas):
    """ Matplotlib Figure Widget  """

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        # create Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)    # explicite call of super constructor
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
        self.setMinimumSize(200, 200)

    def plot(self,X,Y,mode, clear=False):
        if clear: self.axes.clear()
        self.axes.plot(X,Y,mode)
        try:
            self.fig.canvas.draw()
        except Exception:
            time.sleep(0.5)
            self.fig.canvas.draw()

# ------------------------------------------------------------------------------------------
# --- class ImageGalleryView(QSplitter) ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageGalleryView(QSplitter):
    """ 
        ImageGallery(QSplitter)

        +-------------------------------------------+
        | +----+ +----+ +----+ +----+ +----+ +----+ | \
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |   >   GridLayout
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ |  |
        | |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |ImgW| |  |
        | +----+ +----+ +----+ +----+ +----+ +----+ | /
        +-------------------------------------------+  <    splitter
        | [<] [1x1][3x2][6x4][9x6][page number] [>] |       [pushButton] HorizontalLayout
        +-------------------------------------------+

    """
    def __init__(self,controller_=None,shapeMode=None):
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.__init__(",")")

        super().__init__(Qt.Vertical)

        self.controller= controller_
        self.shapeMode = controller.GalleryMode._3x2 if not shapeMode else shapeMode    # default display mode 
        self.pageNumber =0

        self.imagesControllers = []

        self.images = QFrame()
        self.images.setFrameShape(QFrame.StyledPanel)
        self.imagesLayout = QGridLayout()
        self.images.setLayout(self.imagesLayout)

        self.buildGridLayoutWidgets()

        self.previousPageButton =   QPushButton('<')
        self.previousPageButton.clicked.connect(self.controller.callBackButton_previousPage)
        self._1x1Button =           QPushButton('1x1')
        self._1x1Button.clicked.connect(self.controller.callBackButton_1x1)
        self._2x1Button =           QPushButton('2x1')
        self._2x1Button.clicked.connect(self.controller.callBackButton_2x1)
        self._3x2Button =           QPushButton('3x2')
        self._3x2Button.clicked.connect(self.controller.callBackButton_3x2)
        self._6x4Button =           QPushButton('6x4')
        self._6x4Button.clicked.connect(self.controller.callBackButton_6x4)
        self._9x6Button =           QPushButton('9x6')
        self._9x6Button.clicked.connect(self.controller.callBackButton_9x6)
        self.nextPageButton =       QPushButton('>')
        self.nextPageButton.clicked.connect(self.controller.callBackButton_nextPage)

        self.pageNumberLabel = QLabel(str(self.pageNumber)+"/ ...")

        self.buttons = QWidget()
        self.buttonsLayout = QHBoxLayout()
        self.buttons.setLayout(self.buttonsLayout)
        self.buttonsLayout.addWidget(self.previousPageButton)
        self.buttonsLayout.addWidget(self._1x1Button)
        self.buttonsLayout.addWidget(self._2x1Button)
        self.buttonsLayout.addWidget(self._3x2Button)
        self.buttonsLayout.addWidget(self._6x4Button)
        self.buttonsLayout.addWidget(self._9x6Button)
        self.buttonsLayout.addWidget(self.nextPageButton)

        self.buttonsLayout.addWidget(self.pageNumberLabel)

        self.addWidget(self.images)
        self.addWidget(self.buttons)
        self.setSizes([1525,82])

    def currentPage(self): return self.pageNumber

    def changePageNumber(self,step):
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.changePageNumber(",step,")")

        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode)*controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes)-1)//nbImagePerPage) + 1

        if len(self.controller.model.processPipes) > 0 :

            oldPageNumber = self.pageNumber
            if (self.pageNumber+step) > maxPage-1:
                self.pageNumber = 0
            elif (self.pageNumber+step) <0:
                self.pageNumber = maxPage-1
            else:
                self.pageNumber  = self.pageNumber+step
            self.updateImages()
            self.controller.model.loadPage(self.pageNumber)
            if pref.verbose: print(" [VIEW] >> ImageGalleryView.changePageNumber(currentPage:",self.pageNumber," | max page:",maxPage,")")

    def updateImages(self):
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.updateImages(",")")

        """ update images content """
        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode)*controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes)-1)//nbImagePerPage) + 1
        
        index=0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)): 
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                # get image controllers                                                                                         
                iwc = self.imagesControllers[index]
                iwc.view.setPixmap(ImageWidgetView.emptyImageColorData())                                                 
                index +=1                                                                                                                                                                                                           
        self.pageNumberLabel.setText(str(self.pageNumber)+"/"+str(maxPage-1))

    def updateImage(self, idx, processPipe, filename):
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.updateImage(",")")
        imageWidgetController = self.imagesControllers[idx]                                 
        imageWidgetController.setImage(processPipe.getImage())
        self.controller.parent.statusBar().showMessage("loading of image "+filename+" done!")

    def resetGridLayoutWidgets(self):
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.resetGridLayoutWidgets(",")")

        for w in self.imagesControllers:
            self.imagesLayout.removeWidget(w.view)
            w.view.deleteLater()
        self.imagesControllers = []

    def buildGridLayoutWidgets(self):
        if pref.verbose: print(" [VIEW] >> ImageGalleryView.buildGridLayoutWidgets(",")")

        imageIndex = 0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)): 
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                iwc = controller.ImageWidgetController(id=imageIndex)
                self.imagesControllers.append(iwc)
                self.imagesLayout.addWidget(iwc.view,i,j)
                imageIndex +=1

    def wheelEvent(self, event):
        if pref.verbose: print(" [EVENT] >> ImageGalleryView.wheelEvent(",")")

        if event.angleDelta().y() < 0 :     
            self.changePageNumber(+1)
            if self.shapeMode == controller.GalleryMode._1x1 : 
                self.controller.selectImage(0)

        else :                              
            self.changePageNumber(-1)
            if self.shapeMode == controller.GalleryMode._1x1 : 
                self.controller.selectImage(0)
        event.accept()

    def mousePressEvent(self,event):
        if pref.verbose: print(" [EVENT] >> ImageGalleryView.mousePressEvent(",")")

        # self.childAt(event.pos()) return QLabel .parent() should be ImageWidget object
        if isinstance(self.childAt(event.pos()).parent(),ImageWidgetView):
            id = self.childAt(event.pos()).parent().controller.id()
        else:
            id = -1

        if id != -1: # an image is clicked select it!
            self.controller.selectImage(id)
        event.accept()
# ------------------------------------------------------------------------------------------
# --- class AppView(QMainWindow) -----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AppView(QMainWindow):
    """ 
        MainWindow(Vue)
    """
    def __init__(self, _controller = None, shapeMode=None, HDRcontroller=None):
        super().__init__()
        # --------------------
        scale = 0.8
        # --------------------   
        # attributes
        self.controller = _controller
        self.setWindowGeometry(scale=scale)
        self.setWindowTitle('uHDR - Rémi Cozot (c) 2020-2021')      # title  
        self.statusBar().showMessage('Welcome to uHDR!')         # status bar

        self.topContainer = QWidget()
        self.topLayout = QHBoxLayout()

        self.imageGalleryController = controller.ImageGalleryController(self)

        self.topLayout.addWidget(self.imageGalleryController.view)

        self.topContainer.setLayout(self.topLayout)
        self.setCentralWidget(self.topContainer)
        # ----------------------------------
        self.dock = controller.MultiDockController(self, HDRcontroller)
        self.addDockWidget(Qt.RightDockWidgetArea,self.dock.view)
        self.resizeDocks([self.dock.view],[int(self.controller.screenSize[0].width()*scale//4)],Qt.Horizontal)

        # ----------------------------------
        # build menu
        self.buildFileMenu()
        self.buildDockMenu()
        self.buildDisplayHDR()
        self.buildExport()
        self.buildPreferences()
    # ------------------------------------------------------------------------------------------
    def getImageGalleryController(self): return self.imageGalleryController
    # ------------------------------------------------------------------------------------------
    def resizeEvent(self, event): super().resizeEvent(event)
    # ------------------------------------------------------------------------------------------
    def setWindowGeometry(self, scale=0.8):
        displayCoord = QDesktopWidget().screenGeometry(1)
        if len(self.controller.screenSize) > 1:
            width, height = self.controller.screenSize[1].width(), self.controller.screenSize[1].height()
        else:
            width, height = self.controller.screenSize[0].width(), self.controller.screenSize[0].height()

        self.setGeometry(displayCoord.left(), displayCoord.top()+50, math.floor(width*scale), math.floor(height*scale))
        self.showMaximized()
    # ------------------------------------------------------------------------------------------
    def buildFileMenu(self):
        menubar = self.menuBar()# get menubar
        fileMenu = menubar.addMenu('&File')# file menu

        selectDir = QAction('&Select directory', self)        
        selectDir.setShortcut('Ctrl+O')
        selectDir.setStatusTip('[File] select a directory')
        selectDir.triggered.connect(self.controller.callBackSelectDir)
        fileMenu.addAction(selectDir)

        selectSave = QAction('&Save', self)        
        selectSave.setShortcut('Ctrl+S')
        selectSave.setStatusTip('[File] saving processpipe metadata')
        selectSave.triggered.connect(self.controller.callBackSave)
        fileMenu.addAction(selectSave)

        quit = QAction('&Quit', self)        
        quit.setShortcut('Ctrl+Q')
        quit.setStatusTip('[File] saving updates and quit')
        quit.triggered.connect(self.controller.callBackQuit)
        fileMenu.addAction(quit)
    # ------------------------------------------------------------------------------------------    
    def buildPreferences(self):
        """build preferences menu

            Args:

            Returns:

        """

        menubar = self.menuBar()# get menubar
        prefMenu = menubar.addMenu('&Preferences')# file menu

        displayList = pref.getHDRdisplays().keys()

        # function for callback
        def cbd(tag): 
            pref.setHDRdisplay(tag)
            self.statusBar().showMessage("swithcing HDR Display to: "+tag+"!")
            self.menuExport.setText('&Export to '+pref.getHDRdisplay()['tag'])
            self.menuExportAll.setText('&Export All to '+pref.getHDRdisplay()['tag'])        

        prefDisplays = []
        for i,d in enumerate(displayList):
            if d != 'none':
                prefDisplay = QAction('&Set display to '+d, self) 
                p_cbd = functools.partial(cbd, d)
                prefDisplay.triggered.connect(p_cbd)
                prefMenu.addAction(prefDisplay)

    # ------------------------------------------------------------------------------------------
    def buildDisplayHDR(self):
        menubar = self.menuBar()# get menubar
        displayHDRmenu = menubar.addMenu('&Display HDR')# file menu

        displayHDR = QAction('&Display HDR image', self)        
        displayHDR.setShortcut('Ctrl+H')
        displayHDR.setStatusTip('[Display HDR] display HDR image')
        displayHDR.triggered.connect(self.controller.callBackDisplayHDR)
        displayHDRmenu.addAction(displayHDR)
        # ------------------------------------
        displayHDR = QAction('&Compare raw and edited HDR image', self)        
        displayHDR.setShortcut('Ctrl+C')
        displayHDR.setStatusTip('[Display HDR] compare raw HDR image and edited one')
        displayHDR.triggered.connect(self.controller.callBackCompareRawEditedHDR)
        displayHDRmenu.addAction(displayHDR)
        # ------------------------------------
        closeHDR = QAction('&reset HDR display', self)        
        closeHDR.setShortcut('Ctrl+K')
        closeHDR.setStatusTip('[Display HDR] reset HDR window')
        closeHDR.triggered.connect(self.controller.callBackCloseDisplayHDR)
        displayHDRmenu.addAction(closeHDR)
    # ------------------------------------------------------------------------------------------  
    def buildExport(self):
        menubar = self.menuBar()# get menubar
        exportHDR = menubar.addMenu('&Export HDR image')# file menu

        self.menuExport = QAction('&Export to '+pref.getHDRdisplay()['tag'], self)        
        self.menuExport.setShortcut('Ctrl+X')
        self.menuExport.setStatusTip('[Export HDR image] save HDR image file for HDR display')
        self.menuExport.triggered.connect(self.controller.callBackExportHDR)
        exportHDR.addAction(self.menuExport)

        self.menuExportAll = QAction('&Export All to '+pref.getHDRdisplay()['tag'], self)        
        self.menuExportAll.setShortcut('Ctrl+Y')
        self.menuExportAll.setStatusTip('[Export all HDR images] save HDR image files for HDR display.')
        self.menuExportAll.triggered.connect(self.controller.callBackExportAllHDR)
        exportHDR.addAction(self.menuExportAll)
    # ------------------------------------------------------------------------------------------        
    def buildDockMenu(self):
        menubar = self.menuBar()# get menubar
        dockMenu = menubar.addMenu('&Dock')# file menu

        info = QAction('&Info. and Metadata', self)        
        info.setShortcut('Ctrl+I')
        info.setStatusTip('[Dock] image information dock')
        info.triggered.connect(self.dock.activateINFO)
        dockMenu.addAction(info)

        edit = QAction('&Edit', self)        
        edit.setShortcut('Ctrl+E')
        edit.setStatusTip('[Dock] image editing dock')
        edit.triggered.connect(self.dock.activateEDIT)
        dockMenu.addAction(edit)

        iqa = QAction('&Image Aesthetics', self)        
        iqa.setShortcut('Ctrl+A')
        iqa.setStatusTip('[Dock] image aesthetics dock')
        iqa.triggered.connect(self.dock.activateMIAM)
        dockMenu.addAction(iqa)
    # ------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        if pref.verbose: print(" [CB] >> AppView.closeEvent()>> ... closing")
        self.imageGalleryController.save()
        self.controller.hdrDisplay.close()
# ------------------------------------------------------------------------------------------
# --- class ImageInfoView(QSplitter) -------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageInfoView(QSplitter):
    def __init__(self, _controller):
        if pref.verbose: print(" [VIEW] >> ImageInfoView.__init__(",")")

        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QFormLayout()

        # ---------------------------------------------------
        self.imageName =            AdvanceLineEdit(" name:", " ........ ",             self.layout, callBack=None)
        self.imagePath =            AdvanceLineEdit(" path:", " ........ ",             self.layout, callBack=None)
        self.imageSize =            AdvanceLineEdit(" size (pixel):", ".... x .... ",   self.layout, callBack=None)
        self.imageDynamicRange =    AdvanceLineEdit(" dynamic range (f-stops)", " ........ ", self.layout, callBack=None)
        self.colorSpace =           AdvanceLineEdit(" color space:", " ........ ",      self.layout, callBack=None)
        self.imageType =            AdvanceLineEdit(" type:", " ........ ",             self.layout, callBack=None)                     
        self.imageBPS =             AdvanceLineEdit(" bits per sample:", " ........ ",  self.layout, callBack=None)
        self.imageExpoTime =        AdvanceLineEdit(" exposure time:", " ........ ",    self.layout, callBack=None)
        self.imageFNumber =         AdvanceLineEdit("f-number:", " ........ ",          self.layout, callBack=None)          
        self.imageISO =             AdvanceLineEdit(" ISO:", " ........ ",              self.layout, callBack=None)           
        self.imageCamera =          AdvanceLineEdit(" camera:", " ........ ",           self.layout, callBack=None)      
        self.imageSoftware =        AdvanceLineEdit(" software:", " ........ ",         self.layout, callBack=None)      
        self.imageLens =            AdvanceLineEdit(" lens:", " ........ ",             self.layout, callBack=None)
        self.imageFocalLength =     AdvanceLineEdit(" focal length:", " ........ ",     self.layout, callBack=None)
        # ---------------------------------------------------
        # ---------------------------------------------------
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        self.layout.addRow(line)
        # ---------------------------------------------------
        #  user defined tags
        # --------------------------------------------------
        userDefinedTags = hdrCore.metadata.tags()
        tagRootName = userDefinedTags.getTagsRootName()
        listOfTags = userDefinedTags.tags[tagRootName]
        self.userDefinedTags = []
        for tagGroup in listOfTags:
            groupKey = list(tagGroup.keys())[0]
            tagLeafs = tagGroup[groupKey]
            for tag in tagLeafs.items():
                self.userDefinedTags.append( AdvanceCheckBox(self,groupKey, tag[0], False, self.layout))
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            self.layout.addRow(line)
        # --------------------------------------------------


        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60,40])

    def setProcessPipe(self,processPipe): 
        image_ = processPipe.getImage()
        # ---------------------------------------------------
        if pref.verbose: print(" [VIEW] >> ImageInfoView.setImage(",image_.name,")")
        if image_.metadata.metadata['filename'] != None: self.imageName.setText(image_.metadata.metadata['filename'])
        else: self.imageName.setText(" ........ ")
        if image_.metadata.metadata['path'] != None: self.imagePath.setText(image_.metadata.metadata['path'])
        else: self.imagePath.setText(" ........ ")
        if image_.metadata.metadata['exif']['Image Width'] != None: self.imageSize.setText(str(image_.metadata.metadata['exif']['Image Width'])+" x "+ str(image_.metadata.metadata['exif']['Image Height']))
        else: self.imageSize.setText(" ........ ")
        if image_.metadata.metadata['exif']['Dynamic Range (stops)'] != None: self.imageDynamicRange.setText(str(image_.metadata.metadata['exif']['Dynamic Range (stops)']))
        else: self.imageDynamicRange.setText(" ........ ")
        if image_.metadata.metadata['exif']['Color Space'] != None: self.colorSpace.setText(image_.metadata.metadata['exif']['Color Space'])
        else: self.imageName.setText(" ........ ")
        if image_.type != None: self.imageType.setText(str(image_.type))       
        else: self.colorSpace.setText(" ........ ")
        if image_.metadata.metadata['exif']['Bits Per Sample'] != None: self.imageBPS.setText(str(image_.metadata.metadata['exif']['Bits Per Sample']))
        else: self.imageBPS.setText(" ........ ")
        if image_.metadata.metadata['exif']['Exposure Time'] != None: self.imageExpoTime.setText(str(image_.metadata.metadata['exif']['Exposure Time'][0])+" / " + str(image_.metadata.metadata['exif']['Exposure Time'][1]))
        else: self.imageExpoTime.setText(" ........ ")
        if image_.metadata.metadata['exif']['F Number'] != None: self.imageFNumber.setText(str(image_.metadata.metadata['exif']['F Number'][0]))    
        else: self.imageFNumber.setText(" ........ ")
        if image_.metadata.metadata['exif']['ISO'] != None: self.imageISO.setText(str(image_.metadata.metadata['exif']['ISO']))       
        else: self.imageISO.setText(" ........ ")
        if image_.metadata.metadata['exif']['Camera'] != None: self.imageCamera.setText(image_.metadata.metadata['exif']['Camera'])      
        else: self.imageCamera.setText(" ........ ")
        if image_.metadata.metadata['exif']['Software'] != None: self.imageSoftware.setText(image_.metadata.metadata['exif']['Software'])      
        else: self.imageSoftware.setText(" ........ ")
        if image_.metadata.metadata['exif']['Lens'] != None: self.imageLens.setText(image_.metadata.metadata['exif']['Lens'])
        else: self.imageLens.setText(" ........ ")
        if image_.metadata.metadata['exif']['Focal Length'] != None: self.imageFocalLength.setText(str(image_.metadata.metadata['exif']['Focal Length'][0]))
        else: self.imageFocalLength.setText(" ........ ")
        # ---------------------------------------------------
        self.controller.callBackActive = False
        # ---------------------------------------------------
        tagRootName = image_.metadata.otherTags.getTagsRootName()
        listOfTags = image_.metadata.metadata[tagRootName]

        for i,tagGroup in enumerate(listOfTags):
            groupKey = list(tagGroup.keys())[0]
            tagLeafs = tagGroup[groupKey]
            for tag in tagLeafs.items():
                # find advanced checkbox
                for acb in self.userDefinedTags:
                    if (acb.rightText ==tag[0] ) and (acb.leftText== groupKey):
                        on_off = image_.metadata.metadata[tagRootName][i][groupKey][tag[0]]
                        on_off = on_off if on_off else False
                        acb.setState(on_off)
                        break  
        # ---------------------------------------------------        
        self.controller.callBackActive = True
        # ---------------------------------------------------
        return self.imageWidgetController.setImage(image_)

    def metadataChange(self,metaGroup,metaTag, on_off): 
        if self.controller.callBackActive: self.controller.metadataChange(metaGroup,metaTag, on_off)
# ------------------------------------------------------------------------------------------
# --- class AdvanceLineEdit(object) --------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceLineEdit(object):
    def __init__(self, labelName, defaultText, layout, callBack=None):
        self.label = QLabel(labelName)
        self.lineEdit =QLineEdit(defaultText)
        if callBack: self.lineEdit.textChanged.connect(callBack)
        layout.addRow(self.label,self.lineEdit)

    def setText(self, txt): self.lineEdit.setText(txt)
# ------------------------------------------------------------------------------------------
# --- class AdvanceCheckBox(object) --------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceCheckBox(object):
    def __init__(self, parent, leftText, rightText, defaultValue, layout):
        self.parent = parent

        self.leftText = leftText
        self.rightText = rightText

        self.label = QLabel(leftText)
        self.checkbox =QCheckBox(rightText)
        self.checkbox.toggled.connect(self.toggled)
        layout.addRow(self.label,self.checkbox)

    def setState(self, on_off): self.checkbox.setChecked(on_off)

    def toggled(self): self.parent.metadataChange(self.leftText, self.rightText, self.checkbox.isChecked())
# ------------------------------------------------------------------------------------------
# --- class EditImageView(QSplitter) -------------------------------------------------------
# ------------------------------------------------------------------------------------------
class EditImageView(QSplitter):

    def __init__(self, _controller, build=False):
        if pref.verbose: print(" [VIEW] >> EditImageView.__init__(",")")
        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QVBoxLayout()

        # exposure ----------------------
        self.exposure = controller.AdvanceSliderController(self, "exposure",0,(-10,+10),0.25)
        # call back functions
        self.exposure.callBackAutoPush = self.autoExposure
        self.exposure.callBackValueChange = self.changeExposure
        self.layout.addWidget(self.exposure.view)

        # contrast ----------------------
        self.contrast = controller.AdvanceSliderController(self, "contrast",0,(-100,+100),1)
        # call back functions
        self.contrast.callBackAutoPush = self.autoContrast
        self.contrast.callBackValueChange = self.changeContrast
        self.layout.addWidget(self.contrast.view)

        # tonecurve ----------------------
        self.tonecurve = controller.ToneCurveController(self)        
        self.layout.addWidget(self.tonecurve.view)                   

        # mask ----------------------
        self.lightnessmask = controller.LightnessMaskController(self)        
        self.layout.addWidget(self.lightnessmask.view)                   
 
        # saturation ----------------------
        self.saturation = controller.AdvanceSliderController(self, "saturation",0,(-100,+100),1)
        # call back functions
        self.saturation.callBackAutoPush = self.autoSaturation
        self.saturation.callBackValueChange = self.changeSaturation
        self.layout.addWidget(self.saturation.view)

        # color0 ----------------------
        self.colorEditor0 = controller.LchColorSelectorController(self, idName = "colorEditor0")
        self.layout.addWidget(self.colorEditor0.view)

        # color1 ----------------------
        self.colorEditor1 = controller.LchColorSelectorController(self, idName = "colorEditor1")
        self.layout.addWidget(self.colorEditor1.view)

        # color2 ----------------------
        self.colorEditor2 = controller.LchColorSelectorController(self, idName = "colorEditor2")
        self.layout.addWidget(self.colorEditor2.view)

        # color3 ----------------------
        self.colorEditor3 = controller.LchColorSelectorController(self, idName = "colorEditor3")
        self.layout.addWidget(self.colorEditor3.view)

        # color1 ----------------------
        self.colorEditor4 = controller.LchColorSelectorController(self, idName = "colorEditor4")
        self.layout.addWidget(self.colorEditor4.view)

        # auto color selection ----------------------
        # -----
        self.colorEditorsAuto = controller.ColorEditorsAutoController(self,
                                                                      [self.colorEditor0,
                                                                       self.colorEditor1,
                                                                       self.colorEditor2,
                                                                       self.colorEditor3,
                                                                       self.colorEditor4],
                                                                       "saturation")
        self.layout.addWidget(self.colorEditorsAuto.view)
        # -----

        # geometry ----------------------
        self.geometry = controller.GeometryController(self)
        self.layout.addWidget(self.geometry.view)

        # hdr preview ----------------------
        self.hdrPreview = HDRviewerView(self.controller.controllerHDR, build)
        self.controller.controllerHDR.setView(self.hdrPreview)
        self.layout.addWidget(self.hdrPreview)

        # scroll ------------------------------------------------
        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        # adding widgets to self (QSplitter)
        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60,40])

    def setImage(self,image):
        if pref.verbose: print(" [VIEW] >> EditImageView.setImage(",image.name,")")
        return self.imageWidgetController.setImage(image)

    def autoExposure(self):
        if pref.verbose: print(" [CB] >> EditImageView.autoExposure(",")")

        self.controller.autoExposure()
        pass

    def changeExposure(self, value):
        if pref.verbose: print(" [CB] >> EditImageView.changeExposure(",")")

        self.controller.changeExposure(value)
        pass

    def autoContrast(self):
        if pref.verbose: print(" [CB] >> EditImageView.autoContrast(",")")
        pass

    def changeContrast(self, value):
        if pref.verbose: print(" [CB] >> EditImageView.changeContrast(",")")

        self.controller.changeContrast(value)
        pass

    def autoSaturation(self):
        print(" [CB] >> EditImageView.autoSaturation(",")")
        pass

    def changeSaturation(self,value):   ### TO DO
        if pref.verbose:  print(" [CB] >> EditImageView.changeSaturation(",")")
        self.controller.changeSaturation(value)

    def plotToneCurve(self): self.tonecurve.plotCurve()
 
    def setProcessPipe(self, processPipe):
        """ 
            called to initialize EditImageView on image change
            recover parameters of processPipe
            and initialize view components
        """
        if pref.verbose:  print(" [VIEW] >> EditImageView.setProcessPipe(",")")

        # exposure
        # recover value in pipe and restore it
        id = processPipe.getProcessNodeByName("exposure")
        value = processPipe.getParameters(id)
        self.exposure.setValue(value['EV'], callBackActive = False)

        # contrast
        # recover value in pipe and restore it
        id = processPipe.getProcessNodeByName("contrast")
        value = processPipe.getParameters(id)
        self.contrast.setValue(value['contrast'], callBackActive = False)

        # tonecurve
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("tonecurve")
        value = processPipe.getParameters(id)
        self.tonecurve.setValues(value,callBackActive = False)

        # lightnessmask
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("lightnessmask")
        value = processPipe.getParameters(id)
        self.lightnessmask.setValues(value,callBackActive = False)

        # saturation
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("saturation")
        value = processPipe.getParameters(id)
        self.saturation.setValue(value['saturation'], callBackActive = False)

        # colorEditor0
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor0")
        values = processPipe.getParameters(id)
        self.colorEditor0.setValues(values, callBackActive = False)

        # colorEditor1
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor1")
        values = processPipe.getParameters(id)
        self.colorEditor1.setValues(values, callBackActive = False)

        # colorEditor2
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor2")
        values = processPipe.getParameters(id)
        self.colorEditor2.setValues(values, callBackActive = False)

        # colorEditor3
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor3")
        values = processPipe.getParameters(id)
        self.colorEditor3.setValues(values, callBackActive = False)
        
        # colorEditor4
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("colorEditor4")
        values = processPipe.getParameters(id)
        self.colorEditor4.setValues(values, callBackActive = False)
        
        # geometry
        # recover data in pipe and restore it
        id = processPipe.getProcessNodeByName("geometry")
        values = processPipe.getParameters(id)
        self.geometry.setValues(values, callBackActive = False)
# ------------------------------------------------------------------------------------------
# --- class MultiDockView(QDockWidget) -----------------------------------------------------
# ------------------------------------------------------------------------------------------
class MultiDockView(QDockWidget):
    def __init__(self, _controller, HDRcontroller=None):
        if pref.verbose:  print(" [VIEW] >> MultiDockView.__init__(",")")

        super().__init__("Image Edit/Info")
        self.controller = _controller

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
      
        self.childControllers = [
            controller.EditImageController(self, HDRcontroller), 
            controller.ImageInfoController(self), 
            controller.ImageAestheticsController(self)]
            #controller.ImageQualityController(self, HDRcontroller)]
        self.childController = self.childControllers[0]
        self.active = 0
        self.setWidget(self.childController.view)
        self.repaint()
    # ------------------------------------------------------------------------------------------
    def switch(self,nb):
        """
            change active dock
            nb = 0 > editing imag dock
            nb = 1 > image info and metadata dock
            nb = 2 > image aesthetics model 
        """
        if pref.verbose:  print(" [VIEW] >> MultiDockView.switch(",nb,")")

        if nb != self.active:
     
            self.active = (nb)%len(self.childControllers)
            self.childController.view.deleteLater()
            self.childController = self.childControllers[self.active]
            # rebuild view 
            processPipe = self.controller.parent.imageGalleryController.getSelectedProcessPipe()
            self.childController.buildView(processPipe)
            self.setWidget(self.childController.view)
            self.repaint()
    # ------------------------------------------------------------------------------------------
    def setProcessPipe(self, processPipe):
        if pref.verbose:  print(" [VIEW] >> MultiDockView.setProcessPipe(",processPipe.getImage().name,")")
        return self.childController.setProcessPipe(processPipe)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# --- class AdvanceSliderView(QFrame) ------------------------------------------------------
# ------------------------------------------------------------------------------------------
class AdvanceSliderView(QFrame):
    def __init__(self, controller, name,defaultValue, range, step):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = controller
        self.firstrow = QFrame()

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()
        
        self.firstrow.setLayout(self.hbox)

        self.label= QLabel(name)
        self.auto = QPushButton("auto")
        self.editValue = QLineEdit()
        self.editValue.setValidator(QDoubleValidator())
        self.editValue.setText(str(defaultValue))
        self.reset = QPushButton("reset")

        self.hbox.addWidget(self.label)
        self.hbox.addWidget(self.auto)
        self.hbox.addWidget(self.editValue)
        self.hbox.addWidget(self.reset)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(range[0]/step),int(range[1]/step))
        self.slider.setValue(int(defaultValue/step))
        self.slider.setSingleStep(1) 

        self.vbox.addWidget(self.firstrow)
        self.vbox.addWidget(self.slider)

        self.setLayout(self.vbox)

        # callBackFunctions slider/reset/auto
        self.slider.valueChanged.connect(self.controller.sliderChange)
        self.reset.clicked.connect(self.controller.reset)
        self.auto.clicked.connect(self.controller.auto)
# ------------------------------------------------------------------------------------------
# --- class ToneCurveView(QFrame) ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ToneCurveView(QFrame):

    def __init__(self, controller):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)

        self.controller = controller

        self.vbox = QVBoxLayout()

        # figure
        self.curve = FigureWidget(self) 
        self.curve.setMinimumSize(200, 600)

        self.curve.plot([0.0,100],[0.0,100.0],'r--')

        #containers
        
        # zj add for semi-auto curve begin
        self.containerAuto = QFrame() 
        self.hboxAuto = QHBoxLayout() 
        self.containerAuto.setLayout(self.hboxAuto)
        # zj add for semi-auto curve end                                
        self.containerShadows = QFrame()
        self.hboxShadows = QHBoxLayout()
        self.containerShadows.setLayout(self.hboxShadows)
        
        self.containerBlacks = QFrame()
        self.hboxBlacks = QHBoxLayout()
        self.containerBlacks.setLayout(self.hboxBlacks)

        self.containerMediums = QFrame()
        self.hboxMediums = QHBoxLayout()
        self.containerMediums.setLayout(self.hboxMediums)

        self.containerWhites = QFrame()
        self.hboxWhites = QHBoxLayout()
        self.containerWhites.setLayout(self.hboxWhites)

        self.containerHighlights = QFrame()
        self.hboxHighlights = QHBoxLayout()
        self.containerHighlights.setLayout(self.hboxHighlights)

        self.vbox.addWidget(self.curve)
        self.vbox.addWidget(self.containerAuto) #  zj add for semi-auto curve                                                                       
        self.vbox.addWidget(self.containerHighlights)
        self.vbox.addWidget(self.containerWhites)
        self.vbox.addWidget(self.containerMediums)
        self.vbox.addWidget(self.containerBlacks)
        self.vbox.addWidget(self.containerShadows)

        # zj add for semi-auto curve begin
        # autoCurve button
        self.autoCurve = QPushButton("auto")
        self.hboxAuto.addWidget(self.autoCurve)
        self.hboxAuto.setAlignment(Qt.AlignCenter)
        # zj add for semi-auto curve end
        # shadows
        self.labelShadows = QLabel("shadows")
        self.sliderShadows = QSlider(Qt.Horizontal)
        self.sliderShadows.setRange(0,100)
        self.sliderShadows.setValue(int(self.controller.model.default['shadows'][1]))
        self.editShadows = QLineEdit()
        self.editShadows.setText(str(self.controller.model.default['shadows'][1]))
        self.resetShadows = QPushButton("reset")
        self.hboxShadows.addWidget(self.labelShadows)
        self.hboxShadows.addWidget(self.sliderShadows)
        self.hboxShadows.addWidget(self.editShadows)
        self.hboxShadows.addWidget(self.resetShadows)

        # blacks
        self.labelBlacks = QLabel("  blacks  ")
        self.sliderBlacks = QSlider(Qt.Horizontal)
        self.sliderBlacks.setRange(0,100)
        self.sliderBlacks.setValue(int(self.controller.model.default['blacks'][1]))
        self.editBlacks = QLineEdit()
        self.editBlacks.setText(str(self.controller.model.default['blacks'][1]))
        self.resetBlacks = QPushButton("reset")
        self.hboxBlacks.addWidget(self.labelBlacks)
        self.hboxBlacks.addWidget(self.sliderBlacks)
        self.hboxBlacks.addWidget(self.editBlacks)
        self.hboxBlacks.addWidget(self.resetBlacks)

        # mediums
        self.labelMediums = QLabel("mediums")
        self.sliderMediums = QSlider(Qt.Horizontal)
        self.sliderMediums.setRange(0,100)
        self.sliderMediums.setValue(int(self.controller.model.default['mediums'][1]))
        self.editMediums = QLineEdit()
        self.editMediums.setText(str(self.controller.model.default['mediums'][1]))
        self.resetMediums = QPushButton("reset")
        self.hboxMediums.addWidget(self.labelMediums)
        self.hboxMediums.addWidget(self.sliderMediums)
        self.hboxMediums.addWidget(self.editMediums)
        self.hboxMediums.addWidget(self.resetMediums)

        # whites
        self.labelWhites = QLabel("  whites  ")
        self.sliderWhites = QSlider(Qt.Horizontal)
        self.sliderWhites.setRange(0,100)
        self.sliderWhites.setValue(int(self.controller.model.default['whites'][1]))
        self.editWhites = QLineEdit()
        self.editWhites.setText(str(self.controller.model.default['whites'][1]))
        self.resetWhites = QPushButton("reset")
        self.hboxWhites.addWidget(self.labelWhites)
        self.hboxWhites.addWidget(self.sliderWhites)
        self.hboxWhites.addWidget(self.editWhites)
        self.hboxWhites.addWidget(self.resetWhites)

        # highlights
        self.labelHighlights = QLabel("highlights")
        self.sliderHighlights = QSlider(Qt.Horizontal)
        self.sliderHighlights.setRange(0,100)
        self.sliderHighlights.setValue(int(self.controller.model.default['highlights'][1]))
        self.editHighlights = QLineEdit()
        self.editHighlights.setText(str(self.controller.model.default['highlights'][1]))
        self.resetHighlights = QPushButton("reset")
        self.hboxHighlights.addWidget(self.labelHighlights)
        self.hboxHighlights.addWidget(self.sliderHighlights)
        self.hboxHighlights.addWidget(self.editHighlights)
        self.hboxHighlights.addWidget(self.resetHighlights)

        self.setLayout(self.vbox)

        # callBackFunctions slider/reset
        self.sliderShadows.valueChanged.connect(self.sliderShadowsChange)
        self.sliderBlacks.valueChanged.connect(self.sliderBlacksChange)
        self.sliderMediums.valueChanged.connect(self.sliderMediumsChange)
        self.sliderWhites.valueChanged.connect(self.sliderWhitesChange)
        self.sliderHighlights.valueChanged.connect(self.sliderHighlightsChange)

        self.resetShadows.clicked.connect(self.resetShadowsCB)
        self.resetBlacks.clicked.connect(self.resetBlacksCB)
        self.resetMediums.clicked.connect(self.resetMediumsCB)
        self.resetWhites.clicked.connect(self.resetWhitesCB)
        self.resetHighlights.clicked.connect(self.resetHighlightsCB)
        self.autoCurve.clicked.connect(self.controller.autoCurve)  #  zj add for semi-auto curve 
                                                                                         

    def sliderShadowsChange(self):
        if self.controller.callBackActive:
            value = self.sliderShadows.value()
            self.controller.sliderChange("shadows", value)
        pass

    def sliderBlacksChange(self):
        if self.controller.callBackActive:
            value = self.sliderBlacks.value()
            self.controller.sliderChange("blacks", value)
        pass

    def sliderMediumsChange(self):
        if self.controller.callBackActive:
            value = self.sliderMediums.value()
            self.controller.sliderChange("mediums", value)
        pass

    def sliderWhitesChange(self):
        if self.controller.callBackActive:
            value = self.sliderWhites.value()
            self.controller.sliderChange("whites", value)
        pass

    def sliderHighlightsChange(self):
        if self.controller.callBackActive:
            value = self.sliderHighlights.value()
            self.controller.sliderChange("highlights", value)
        pass

    def resetShadowsCB(self):
        if self.controller.callBackActive: self.controller.reset("shadows")

    def resetBlacksCB(self):
        if self.controller.callBackActive: self.controller.reset("blacks")
    
    def resetMediumsCB(self):
        if self.controller.callBackActive: self.controller.reset("mediums")

    def resetWhitesCB(self):
        if self.controller.callBackActive: self.controller.reset("whites")

    def resetHighlightsCB(self):
        if self.controller.callBackActive: self.controller.reset("highlights")
# ------------------------------------------------------------------------------------------
# --- class LightnessMaskView(QGroupBox) ---------------------------------------------------
# ------------------------------------------------------------------------------------------
class LightnessMaskView(QGroupBox):
    def __init__(self, _controller):
        super().__init__("mask lightness")
        #self.setFrameShape(QFrame.StyledPanel)

        self.controller = _controller

        self.hbox = QHBoxLayout()
        self.setLayout(self.hbox)

        self.checkboxShadows = QCheckBox("shadows")
        self.checkboxShadows.setChecked(False)
        self.checkboxBlacks = QCheckBox("blacks")
        self.checkboxBlacks.setChecked(False)
        self.checkboxMediums = QCheckBox("mediums")
        self.checkboxMediums.setChecked(False)
        self.checkboxWhites = QCheckBox("whites")
        self.checkboxWhites.setChecked(False)
        self.checkboxHighlights = QCheckBox("highlights")
        self.checkboxHighlights.setChecked(False)

        self.checkboxShadows.toggled.connect(self.clickShadows)
        self.checkboxBlacks.toggled.connect(self.clickBlacks)
        self.checkboxMediums.toggled.connect(self.clickMediums)
        self.checkboxWhites.toggled.connect(self.clickWhites)
        self.checkboxHighlights.toggled.connect(self.clickHighlights)

        self.hbox.addWidget(self.checkboxShadows)
        self.hbox.addWidget(self.checkboxBlacks)
        self.hbox.addWidget(self.checkboxMediums)
        self.hbox.addWidget(self.checkboxWhites)
        self.hbox.addWidget(self.checkboxHighlights)

    # callbacks
    def clickShadows(self):     
        if self.controller.callBackActive:  self.controller.maskChange("shadows", self.checkboxShadows.isChecked())
    def clickBlacks(self):     
        if self.controller.callBackActive:  self.controller.maskChange("blacks", self.checkboxBlacks.isChecked())
    def clickMediums(self):     
        if self.controller.callBackActive:  self.controller.maskChange("mediums", self.checkboxMediums.isChecked())
    def clickWhites(self):     
        if self.controller.callBackActive:  self.controller.maskChange("whites", self.checkboxWhites.isChecked())
    def clickHighlights(self):     
        if self.controller.callBackActive:  self.controller.maskChange("highlights", self.checkboxHighlights.isChecked())
# ------------------------------------------------------------------------------------------
# --- class HDRviewerView(QFrame) ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class HDRviewerView(QFrame):
    def __init__(self, _controller= None, build = False):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)

        self.controller = _controller

        self.vbox = QVBoxLayout()
        self.hboxUp = QHBoxLayout()
        self.hboxDown = QHBoxLayout()

        self.label= QLabel("hdr preview")
        self.resetButton = QPushButton("reset")
        self.updateButton = QPushButton("update")
        self.compareButton = QPushButton("compare")
        self.autoCheckBox = QCheckBox("auto")
        if build:
            cValue = self.controller.parent.view.dock.view.childControllers[0].model.autoPreviewHDR
            self.autoCheckBox.setChecked(cValue)
        else:
            self.autoCheckBox.setChecked(False)

        self.hboxUpContainer = QFrame()
        self.hboxUpContainer.setLayout(self.hboxUp)
        self.hboxUp.addWidget(self.label)
        self.hboxUp.addWidget(self.resetButton)

        self.hboxDownContainer = QFrame()
        self.hboxDownContainer.setLayout(self.hboxDown)
        self.hboxDown.addWidget(self.autoCheckBox)
        self.hboxDown.addWidget(self.updateButton)
        self.hboxDown.addWidget(self.compareButton)

        self.vbox.addWidget(self.hboxUpContainer)
        self.vbox.addWidget(self.hboxDownContainer)

        self.setLayout(self.vbox)

        self.resetButton.clicked.connect(self.reset)
        self.updateButton.clicked.connect(self.update)
        self.compareButton.clicked.connect(self.compare)
        self.autoCheckBox.toggled.connect(self.auto)

    def reset(self): self.controller.displaySplash()

    def update(self): self.controller.callBackUpdate()

    def compare(self): self.controller.callBackCompare()

    def auto(self): self.controller.callBackAuto(self.autoCheckBox.isChecked())
# ------------------------------------------------------------------------------------------
# --- class LchColorSelectorView(QFrame) ---------------------------------------------------
# ------------------------------------------------------------------------------------------
class LchColorSelectorView(QFrame):
    def __init__(self, _controller, defaultValues=None):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = _controller

        self.vbox = QVBoxLayout()

        self.labelSelector = QLabel("Hue Chroma Lighness color selector")
        # procedural image: Hue bar
        hueBarLch = hdrCore.image.Image.buildLchColorData((75,75), (100,100), (0,360), (20,720), width='h', height='c')
        hueBarRGB = hdrCore.processing.Lch_to_sRGB(hueBarLch,apply_cctf_encoding=True, clip=True)
        self.imageHueController = controller.ImageWidgetController()
        self.imageHueController.view.setMinimumSize(2, 72)
        self.imageHueController.setImage(hueBarRGB)
        hueBar2Lch = hdrCore.image.Image.buildLchColorData((75,75), (100,100), (0,360), (20,720), width='h', height='c')
        hueBar2RGB = hdrCore.processing.Lch_to_sRGB(hueBar2Lch,apply_cctf_encoding=True, clip=True)
        self.imageHueRangeController = controller.ImageWidgetController()
        self.imageHueRangeController.view.setMinimumSize(2, 72)
        self.imageHueRangeController.setImage(hueBarRGB)
        # slider min
        self.sliderHueMin = QSlider(Qt.Horizontal)
        self.sliderHueMin.setRange(0,360)
        self.sliderHueMin.setValue(0)
        self.sliderHueMin.setSingleStep(1)
        # slider max
        self.sliderHueMax = QSlider(Qt.Horizontal)
        self.sliderHueMax.setRange(0,360)
        self.sliderHueMax.setValue(360)
        self.sliderHueMax.setSingleStep(1)

        # procedural image: Saturation bar
        saturationBarLch = hdrCore.image.Image.buildLchColorData((75,75), (0,100), (180,180), (20,720), width='c', height='L')
        saturationBarRGB = hdrCore.processing.Lch_to_sRGB(saturationBarLch,apply_cctf_encoding=True, clip=True)
        self.imageSaturationController = controller.ImageWidgetController()
        self.imageSaturationController.view.setMinimumSize(2, 72)
        self.imageSaturationController.setImage(saturationBarRGB)
        # slider min
        self.sliderChromaMin = QSlider(Qt.Horizontal)
        self.sliderChromaMin.setRange(0,100)
        self.sliderChromaMin.setValue(0)
        self.sliderChromaMin.setSingleStep(1)
        # slider max
        self.sliderChromaMax = QSlider(Qt.Horizontal)
        self.sliderChromaMax.setRange(0,100)
        self.sliderChromaMax.setValue(100)
        self.sliderChromaMax.setSingleStep(1)

        # procedural image:lightness bar
        lightnessBarLch = hdrCore.image.Image.buildLchColorData((0,100), (0,0), (180,180), (20,720), width='L', height='c')
        lightnessBarRGB = hdrCore.processing.Lch_to_sRGB(lightnessBarLch,apply_cctf_encoding=True, clip=True)
        self.imageLightnessController = controller.ImageWidgetController()
        self.imageLightnessController.view.setMinimumSize(2, 72)
        self.imageLightnessController.setImage(lightnessBarRGB)
        # slider min
        self.sliderLightMin = QSlider(Qt.Horizontal)
        self.sliderLightMin.setRange(0,300)
        self.sliderLightMin.setValue(0)
        self.sliderLightMin.setSingleStep(1)        
        # slider max
        self.sliderLightMax = QSlider(Qt.Horizontal)
        self.sliderLightMax.setRange(0,300)
        self.sliderLightMax.setValue(300)
        self.sliderLightMax.setSingleStep(1)

        # editor
        self.labelEditor =      QLabel("color editor: hue shift, exposure, contrast, saturation")

        # hue shift [-180,+180]
        self.frameHueShift = QFrame()
        self.layoutHueShift = QHBoxLayout()
        self.frameHueShift.setLayout(self.layoutHueShift)
        self.sliderHueShift =   QSlider(Qt.Horizontal)
        self.sliderHueShift.setRange(-180,+180)
        self.sliderHueShift.setValue(0)
        self.sliderHueShift.setSingleStep(1) 
        self.valueHueShift = QLineEdit()
        self.valueHueShift.setText(str(0.0))  
        self.layoutHueShift.addWidget(QLabel("hue shift"))
        self.layoutHueShift.addWidget(self.sliderHueShift)
        self.layoutHueShift.addWidget(self.valueHueShift)

        # exposure [-3 , +3]
        self.frameExposure = QFrame()
        self.layoutExposure = QHBoxLayout()
        self.frameExposure.setLayout(self.layoutExposure)
        self.sliderExposure =   QSlider(Qt.Horizontal)
        self.sliderExposure.setRange(-90,+90)
        self.sliderExposure.setValue(0)
        self.sliderExposure.setSingleStep(1) 
        self.valueExposure = QLineEdit()
        self.valueExposure.setText(str(0.0))  
        self.layoutExposure.addWidget(QLabel("exposure"))
        self.layoutExposure.addWidget(self.sliderExposure)
        self.layoutExposure.addWidget(self.valueExposure)

        # contrast [-100 , +100]
        self.frameContrast = QFrame()
        self.layoutContrast = QHBoxLayout()
        self.frameContrast.setLayout(self.layoutContrast)
        self.sliderContrast =   QSlider(Qt.Horizontal)
        self.sliderContrast.setRange(-100,+100)
        self.sliderContrast.setValue(0)
        self.sliderContrast.setSingleStep(1) 
        self.valueContrast = QLineEdit()
        self.valueContrast.setText(str(0.0))  
        self.layoutContrast.addWidget(QLabel("contrast"))
        self.layoutContrast.addWidget(self.sliderContrast)
        self.layoutContrast.addWidget(self.valueContrast)

        # saturation [-100 , +100]
        self.frameSaturation = QFrame()
        self.layoutSaturation = QHBoxLayout()
        self.frameSaturation.setLayout(self.layoutSaturation)
        self.sliderSaturation =   QSlider(Qt.Horizontal)
        self.sliderSaturation.setRange(-100,+100)
        self.sliderSaturation.setValue(0)
        self.sliderSaturation.setSingleStep(1) 
        self.valueSaturation = QLineEdit()
        self.valueSaturation.setText(str(0.0))  
        self.layoutSaturation.addWidget(QLabel("saturation"))
        self.layoutSaturation.addWidget(self.sliderSaturation)
        self.layoutSaturation.addWidget(self.valueSaturation)

        # -----
        self.resetSelection  = QPushButton("reset selection")
        self.resetEdit  = QPushButton("reset edit")
        # -----

        # mask
        self.checkboxMask = QCheckBox("show selection")
        self.checkboxMask.setChecked(False)

        self.vbox.addWidget(self.labelSelector)

        self.vbox.addWidget(self.imageHueController.view)
        self.vbox.addWidget(self.sliderHueMin)
        self.vbox.addWidget(self.sliderHueMax)
        self.vbox.addWidget(self.imageHueRangeController.view)


        self.vbox.addWidget(self.imageSaturationController.view)
        self.vbox.addWidget(self.sliderChromaMin)
        self.vbox.addWidget(self.sliderChromaMax)

        self.vbox.addWidget(self.imageLightnessController.view)
        self.vbox.addWidget(self.sliderLightMin)
        self.vbox.addWidget(self.sliderLightMax)

        # -----
        self.vbox.addWidget(self.resetSelection)
        # -----


        self.vbox.addWidget(self.labelEditor)
        self.vbox.addWidget(self.frameHueShift)
        self.vbox.addWidget(self.frameSaturation)
        self.vbox.addWidget(self.frameExposure)
        self.vbox.addWidget(self.frameContrast)

        self.vbox.addWidget(self.checkboxMask)
        # -----
        self.vbox.addWidget(self.resetEdit)
        # -----
        self.setLayout(self.vbox)

        # callbacks  
        self.sliderHueMin.valueChanged.connect(self.sliderHueChange)
        self.sliderHueMax.valueChanged.connect(self.sliderHueChange)
        self.sliderChromaMin.valueChanged.connect(self.sliderChromaChange)
        self.sliderChromaMax.valueChanged.connect(self.sliderChromaChange)
        self.sliderLightMin.valueChanged.connect(self.sliderLightnessChange)
        self.sliderLightMax.valueChanged.connect(self.sliderLightnessChange)
        self.sliderExposure.valueChanged.connect(self.sliderExposureChange)
        self.sliderSaturation.valueChanged.connect(self.sliderSaturationChange)
        self.sliderContrast.valueChanged.connect(self.sliderContrastChange)
        self.sliderHueShift.valueChanged.connect(self.sliderHueShiftChange)
        self.checkboxMask.toggled.connect(self.checkboxMaskChange)
        # -----
        self.resetSelection.clicked.connect(self.controller.resetSelection)
        self.resetEdit.clicked.connect(self.controller.resetEdit)


    # callbacks
    def sliderHueChange(self):
        hmin = self.sliderHueMin.value()
        hmax = self.sliderHueMax.value()

        # redraw hue range and chroma bar
        hueRangeBarLch = hdrCore.image.Image.buildLchColorData((75,75), (100,100), (hmin,hmax), (20,720), width='h', height='c')
        hueRangeBarRGB = hdrCore.processing.Lch_to_sRGB(hueRangeBarLch,apply_cctf_encoding=True, clip=True)
        self.imageHueRangeController.setImage(hueRangeBarRGB)
        saturationBarLch = hdrCore.image.Image.buildLchColorData((75,75), (0,100), (hmin,hmax), (20,720), width='c', height='L')
        saturationBarRGB = hdrCore.processing.Lch_to_sRGB(saturationBarLch,apply_cctf_encoding=True, clip=True)
        self.imageSaturationController.setImage(saturationBarRGB)

        # call controller
        self.controller.sliderHueChange(hmin,hmax)

    def sliderChromaChange(self):
        vmin = self.sliderChromaMin.value()
        vmax = self.sliderChromaMax.value()
        # call controller
        self.controller.sliderChromaChange(vmin,vmax)

    def sliderLightnessChange(self):
        vmin = self.sliderLightMin.value()/3.0
        vmax = self.sliderLightMax.value()/3.0
        # call controller
        self.controller.sliderLightnessChange(vmin,vmax)

    def sliderExposureChange(self):
        ev = round(self.sliderExposure.value()/30,1)
        # force to 0.1 precision
        self.valueExposure.setText(str(ev))
        self.controller.sliderExposureChange(ev)

    def sliderSaturationChange(self):
        ev = self.sliderSaturation.value()
        self.valueSaturation.setText(str(ev))
        self.controller.sliderSaturationChange(ev)

    def sliderContrastChange(self):
        ev = self.sliderContrast.value()
        self.valueContrast.setText(str(ev))
        self.controller.sliderContrastChange(ev)

    def sliderHueShiftChange(self):
        hs = self.sliderHueShift.value()
        self.valueHueShift.setText(str(hs))
        self.controller.sliderHueShiftChange(hs)

    def checkboxMaskChange(self):
        self.controller.checkboxMaskChange(self.checkboxMask.isChecked())
# ------------------------------------------------------------------------------------------
# --- class GeometryView(QFrame) -----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class GeometryView(QFrame):
    def __init__(self, _controller):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = _controller

        self.vbox = QVBoxLayout()

        # cropping adjustement 
        self.frameCroppingVerticalAdjustement = QFrame()
        self.layoutCroppingVerticalAdjustement = QHBoxLayout()
        self.frameCroppingVerticalAdjustement.setLayout(self.layoutCroppingVerticalAdjustement)
        self.sliderCroppingVerticalAdjustement =   QSlider(Qt.Horizontal)
        self.sliderCroppingVerticalAdjustement.setRange(-100,+100)
        self.sliderCroppingVerticalAdjustement.setValue(0)
        self.sliderCroppingVerticalAdjustement.setSingleStep(1) 
        self.valueCroppingVerticalAdjustement = QLineEdit()
        self.valueCroppingVerticalAdjustement.setText(str(0.0))  
        self.layoutCroppingVerticalAdjustement.addWidget(QLabel("cropping adj."))
        self.layoutCroppingVerticalAdjustement.addWidget(self.sliderCroppingVerticalAdjustement)
        self.layoutCroppingVerticalAdjustement.addWidget(self.valueCroppingVerticalAdjustement)

        # rotation 
        self.frameRotation = QFrame()
        self.layoutRotation = QHBoxLayout()
        self.frameRotation.setLayout(self.layoutRotation)
        self.sliderRotation =   QSlider(Qt.Horizontal)
        self.sliderRotation.setRange(-60,+60)
        self.sliderRotation.setValue(0)
        self.sliderRotation.setSingleStep(1) 
        self.valueRotation = QLineEdit()
        self.valueRotation.setText(str(0.0))  
        self.layoutRotation.addWidget(QLabel("rotation"))
        self.layoutRotation.addWidget(self.sliderRotation)
        self.layoutRotation.addWidget(self.valueRotation)

        self.vbox.addWidget(self.frameCroppingVerticalAdjustement)
        self.vbox.addWidget(self.frameRotation)

        self.setLayout(self.vbox)

        self.sliderCroppingVerticalAdjustement.valueChanged.connect(self.sliderCroppingVerticalAdjustementChange)
        self.sliderRotation.valueChanged.connect(self.sliderRotationChange)

    # callbacks
    def sliderCroppingVerticalAdjustementChange(self):
        v = self.sliderCroppingVerticalAdjustement.value()
        self.valueCroppingVerticalAdjustement.setText(str(v))
        # call controller
        self.controller.sliderCroppingVerticalAdjustementChange(v)

    def sliderRotationChange(self):
        v = self.sliderRotation.value()/6
        self.valueRotation.setText(str(v))
        # call controller
        self.controller.sliderRotationChange(v)
# ------------------------------------------------------------------------------------------
# --- class AestheticsImageView(QFrame) ----------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageAestheticsView(QSplitter):
    """class AestheticsImageView(QSplitter): view of AestheticsImageController
    """
    def __init__(self, _controller, build=False):
        if pref.verbose: print(" [VIEW] >> AestheticsImageView.__init__(",")")
        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QVBoxLayout()

        # --------------- color palette: node selector(node name), color number, palette image.
        self.labelColorPalette = QLabel("color palette")
        self.labelNodeSelector = QLabel("process output:")
        self.nodeSelector = QComboBox(self)

        # recover process nodes names from buildProcessPipe
        processNodeNameList = []
        emptyProcessPipe = model.EditImageModel.buildProcessPipe()
        for node in emptyProcessPipe.processNodes: processNodeNameList.append(node.name)

        # add 'output' at the end to help user
        processNodeNameList.append('output')

        self.nodeSelector.addItems(processNodeNameList)
        self.nodeSelector.setCurrentIndex(len(processNodeNameList)-1)

        # QSpinBox
        self.labelColorsNumber = QLabel("number of colors:")
        self.nbColors = QSpinBox(self)
        self.nbColors.setRange(2,8)
        self.nbColors.setValue(5)

        self.paletteImageWidgetController = controller.ImageWidgetController()
        imgPalette = hdrCore.aesthetics.Palette('defaultLab5',np.linspace([0,0,0],[100,0,0],5),hdrCore.image.ColorSpace.build('Lab'), hdrCore.image.imageType.SDR).createImageOfPalette()
        self.paletteImageWidgetController.setImage(imgPalette)
        self.paletteImageWidgetController.view.setMinimumSize(40, 10)

        # add widgets to layout
        self.layout.addWidget(self.labelColorPalette)
        self.layout.addWidget(self.labelNodeSelector)
        self.layout.addWidget(self.nodeSelector)
        self.layout.addWidget(self.labelColorsNumber)
        self.layout.addWidget(self.nbColors)
        self.layout.addWidget(self.paletteImageWidgetController.view)

        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        # scroll and etc.
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        # add widget to QSplitter
        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60,40])
        # --------------- composition:
        # --------------- strength line:

    def setProcessPipe(self,processPipe, paletteImg):
        self.imageWidgetController.setImage(processPipe.getImage())
        self.paletteImageWidgetController.setImage(paletteImg)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ColorEditorsAutoView(QPushButton):
    def __init__(self,controller):
        super().__init__("auto color selection [! reset edit]")
        self.controller = controller

        self.clicked.connect(self.controller.auto)
# ------------------------------------------------------------------------------------------
