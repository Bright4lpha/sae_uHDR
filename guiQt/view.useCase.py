# ------------------------------------------------------------------------------------------
# uHDR project 2020
# ------------------------------------------------------------------------------------------
# author: remi.cozot@univ-littoral.fr
# ------------------------------------------------------------------------------------------
# Qt import
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow, QSplitter, QFrame, QDockWidget
from PyQt5.QtWidgets import QSplitter, QFrame, QSlider, QRadioButton
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QLayout, QScrollArea, QFormLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit,QLineEdit, QRadioButton
from PyQt5.QtWidgets import QAction
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtWidgets 

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from datetime import datetime

import numpy as np
import hdrCore.image 
import math, enum

from . import controller
# ------------------------------------------------------------------------------------------
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
        ##
        #t_begin = datetime.now()
        ##
        qImg = QImage((colorData*255).astype(np.uint8), width, height, bytesPerLine, QImage.Format_RGB888) # QImage
        self.imagePixmap = QPixmap.fromImage(qImg)
        self.resize()
        ###
        ##t_end = datetime.now()
        ##print(" >> ImageWidgetView.setPixmap()","[",t_end-t_begin,"ms]")
        ###
        return self.imagePixmap

    def setQPixmap(self, qPixmap):
        self.imagePixmap = qPixmap
        self.resize()

    def emptyImageColorData(): return np.ones((90,160,3))*(220/255) # create default image (backgorund GUI 240)
# ------------------------------------------------------------------------------------------
class FigureWidget(FigureCanvas):
    """ Matplotlib Figure Widget  """

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        # create Figure
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)    # explicite call of super constructor
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
# ------------------------------------------------------------------------------------------
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
        print(" [VIEW] >> ImageGalleryView.__init__(",")")

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

    def changePageNumber(self,step):
        print(" [VIEW] >> ImageGalleryView.changePageNumber(",step,")")

        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode)*controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes)-1)//nbImagePerPage) + 1

        oldPageNumber = self.pageNumber
        if (self.pageNumber+step) > maxPage-1:
            self.pageNumber = 0
        elif (self.pageNumber+step) <0:
            self.pageNumber = maxPage-1
        else:
            self.pageNumber  = self.pageNumber+step
        self.controller.model.loadPage(self.pageNumber)
        self.updateImages()
        print(" [VIEW] >> ImageGalleryView.changePageNumber(currentPage:",self.pageNumber," | max page:",maxPage,")")

    def updateImages(self):
        print(" [VIEW] >> ImageGalleryView.updateImages(",")")

        """ update images content """
        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode)*controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes)-1)//nbImagePerPage) + 1

        index=0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)): 
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                # get image controllers                                                                                         
                iwc = self.imagesControllers[index]
                idxImage = index+nbImagePerPage*self.pageNumber # image index
                if (idxImage) <len(self.controller.model.processPipes):                                   
                    iwc.setImage(self.controller.model.processPipes[idxImage].getImage())
                else: iwc.view.setPixmap(ImageWidgetView.emptyImageColorData())                                                 
                index +=1                                                                                                       
        self.pageNumberLabel.setText(str(self.pageNumber)+"/"+str(maxPage-1))

    def resetGridLayoutWidgets(self):
        print(" [VIEW] >> ImageGalleryView.resetGridLayoutWidgets(",")")

        for w in self.imagesControllers:
            self.imagesLayout.removeWidget(w.view)
            w.view.deleteLater()
        self.imagesControllers = []

    def buildGridLayoutWidgets(self):
        print(" [VIEW] >> ImageGalleryView.buildGridLayoutWidgets(",")")

        imageIndex = 0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)): 
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                iwc = controller.ImageWidgetController(id=imageIndex)
                self.imagesControllers.append(iwc)
                self.imagesLayout.addWidget(iwc.view,i,j)
                imageIndex +=1

    def wheelEvent(self, event):
        print(" [EVENT] >> ImageGalleryView.wheelEvent(",")")

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
        print(" [EVENT] >> ImageGalleryView.mousePressEvent(",")")

        # self.childAt(event.pos()) return QLabel .parent() should be ImageWidget object
        if isinstance(self.childAt(event.pos()).parent(),ImageWidgetView):
            id = self.childAt(event.pos()).parent().controller.id()
        else:
            id = -1
        if id != -1: # an image is clicked select it!
            self.controller.selectImage(id)
        event.accept()
# ------------------------------------------------------------------------------------------
class AppView(QMainWindow):
    """ 
        MainWindow(Vue)
    """
    def __init__(self, appViewController = None, shapeMode=None):
        super().__init__()
        # --------------------
        scale = 0.8
        # --------------------   
        # attributes
        self.controller = appViewController
        self.setWindowGeometry(scale=scale)
        self.setWindowTitle('uHDR - Rémi Cozot (c) 2020')                      # title  
        self.statusBar().showMessage('Welcome to hdrCore!')                       # status bar

        self.topContainer = QWidget()
        self.topLayout = QHBoxLayout()

        self.imageGalleryController = controller.ImageGalleryController(self)

        self.topLayout.addWidget(self.imageGalleryController.view)

        self.topContainer.setLayout(self.topLayout)
        self.setCentralWidget(self.topContainer)
        # ----------------------------------
        #self.dock = MultiDockView(None)
        self.dock = controller.MultiDockController(self)
        self.addDockWidget(Qt.RightDockWidgetArea,self.dock.view)
        self.resizeDocks([self.dock.view],[self.controller.screenSize[0].width()*scale//4],Qt.Horizontal)

        # ----------------------------------
        # build menu
        self.buildFileMenu()
        self.buildDockMenu()
        self.buildDisplayHDR()
        self.buildInfoMenu()
    # ------------------------------------------------------------------------------------------
    def resizeEvent(self, event): super().resizeEvent(event)
    # ------------------------------------------------------------------------------------------
    def setWindowGeometry(self, scale=0.8):
        width, height = self.controller.screenSize[0].width(), self.controller.screenSize[0].height()
        self.setGeometry(0, 0, math.floor(width*scale), math.floor(height*scale))
    # ------------------------------------------------------------------------------------------
    def buildFileMenu(self):
        menubar = self.menuBar()# get menubar
        fileMenu = menubar.addMenu('&File')# file menu
        selectDir = QAction('&Select directory', self)        
        selectDir.setShortcut('Ctrl+O')
        selectDir.setStatusTip('Select a directory')
        selectDir.triggered.connect(self.controller.callBackSelectDir)
        fileMenu.addAction(selectDir)
    # ------------------------------------------------------------------------------------------    
    def buildInfoMenu(self):
        menubar = self.menuBar()# get menubar
        helpMenu = menubar.addMenu('&Help')# file menu
        debug = QAction('&Debug', self)        
        debug.setShortcut('Ctrl+D')
        debug.setStatusTip('[DEBUG] info.')
        debug.triggered.connect(self.controller.callBackInfo)
        helpMenu.addAction(debug)
# ------------------------------------------------------------------------------------------
    def buildDisplayHDR(self):
        menubar = self.menuBar()# get menubar
        displayHDRmenu = menubar.addMenu('&Display HDR')# file menu
        displayHDR = QAction('&HDR display', self)        
        displayHDR.setShortcut('Ctrl+H')
        displayHDR.setStatusTip('[Display HDR] opening HDR windows')
        displayHDR.triggered.connect(self.controller.callBackDisplayHDR)
        displayHDRmenu.addAction(displayHDR)
        # ------------------------------------
        closeHDR = QAction('&HDR close', self)        
        closeHDR.setShortcut('Ctrl+K')
        closeHDR.setStatusTip('[Display HDR] close HDR windows')
        closeHDR.triggered.connect(self.controller.callBackCloseDisplayHDR)
        displayHDRmenu.addAction(closeHDR)
# ------------------------------------------------------------------------------------------        
    def buildDockMenu(self):
        menubar = self.menuBar()# get menubar
        dockMenu = menubar.addMenu('&Dock')# file menu

        info = QAction('&Info/Metadata', self)        
        info.setShortcut('Ctrl+I')
        info.setStatusTip('Image info.')
        info.triggered.connect(self.dock.switch)
        dockMenu.addAction(info)

        edit = QAction('&Edit', self)        
        edit.setShortcut('Ctrl+E')
        edit.setStatusTip('Edit image')
        edit.triggered.connect(self.dock.switch)
        dockMenu.addAction(edit)
# ------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        print(" [CB] >> AppView.closeEvent()>> ... closing")
        self.imageGalleryController.save()
# ------------------------------------------------------------------------------------------
class ImageInfoView(QSplitter):
    def __init__(self, _controller):
        print(" [VIEW] >> ImageInfoView.__init__(",")")

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

    def setImage(self,image): 
        print(" [VIEW] >> ImageInfoView.setImage(",image.name,")")
        if image.metadata.metadata['filename'] != None: self.imageName.setText(image.metadata.metadata['filename'])
        else: self.imageName.setText(" ........ ")
        if image.metadata.metadata['path'] != None: self.imagePath.setText(image.metadata.metadata['path'])
        else: self.imagePath.setText(" ........ ")
        if image.metadata.metadata['exif']['Image Width'] != None: self.imageSize.setText(str(image.metadata.metadata['exif']['Image Width'])+" x "+ str(image.metadata.metadata['exif']['Image Height']))
        else: self.imageSize.setText(" ........ ")
        if image.metadata.metadata['exif']['Dynamic Range (stops)'] != None: self.imageDynamicRange.setText(str(image.metadata.metadata['exif']['Dynamic Range (stops)']))
        else: self.imageDynamicRange.setText(" ........ ")
        if image.metadata.metadata['exif']['Color Space'] != None: self.colorSpace.setText(image.metadata.metadata['exif']['Color Space'])
        else: self.imageName.setText(" ........ ")
        if image.type != None: self.imageType.setText(str(image.type))       
        else: self.colorSpace.setText(" ........ ")
        if image.metadata.metadata['exif']['Bits Per Sample'] != None: self.imageBPS.setText(str(image.metadata.metadata['exif']['Bits Per Sample']))
        else: self.imageBPS.setText(" ........ ")
        if image.metadata.metadata['exif']['Exposure Time'] != None: self.imageExpoTime.setText(str(image.metadata.metadata['exif']['Exposure Time'][0])+" / " + str(image.metadata.metadata['exif']['Exposure Time'][1]))
        else: self.imageExpoTime.setText(" ........ ")
        if image.metadata.metadata['exif']['F Number'] != None: self.imageFNumber.setText(str(image.metadata.metadata['exif']['F Number'][0]))    
        else: self.imageFNumber.setText(" ........ ")
        if image.metadata.metadata['exif']['ISO'] != None: self.imageISO.setText(str(image.metadata.metadata['exif']['ISO']))       
        else: self.imageISO.setText(" ........ ")
        if image.metadata.metadata['exif']['Camera'] != None: self.imageCamera.setText(image.metadata.metadata['exif']['Camera'])      
        else: self.imageCamera.setText(" ........ ")
        if image.metadata.metadata['exif']['Software'] != None: self.imageSoftware.setText(image.metadata.metadata['exif']['Software'])      
        else: self.imageSoftware.setText(" ........ ")
        if image.metadata.metadata['exif']['Lens'] != None: self.imageLens.setText(image.metadata.metadata['exif']['Lens'])
        else: self.imageLens.setText(" ........ ")
        if image.metadata.metadata['exif']['Focal Length'] != None: self.imageFocalLength.setText(str(image.metadata.metadata['exif']['Focal Length'][0]))
        else: self.imageFocalLength.setText(" ........ ")
        # ---------------------------------------------------





        return self.imageWidgetController.setImage(image)
# ------------------------------------------------------------------------------------------
class AdvanceLineEdit(object):
    def __init__(self, labelName, defaultText, layout, callBack=None):
        self.label = QLabel(labelName)
        self.lineEdit =QLineEdit(defaultText)
        if callBack: self.lineEdit.textChanged.connect(callBack)
        layout.addRow(self.label,self.lineEdit)

    def setText(self, txt): self.lineEdit.setText(txt)
# ------------------------------------------------------------------------------------------
class EditImageView(QSplitter):
    def __init__(self, _controller):
        print(" [VIEW] >> EditImageView.__init__(",")")
        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QVBoxLayout()

        self.exposure = controller.AdvanceSliderController(self, "exposure",0,(-10,+10),0.25)
        # call back functions
        self.exposure.callBackAutoPush = self.autoExposure
        self.exposure.callBackValueChange = self.changeExposure
        self.layout.addWidget(self.exposure.view)

        self.contrast = controller.AdvanceSliderController(self, "contrast",0,(-100,+100),1)
        # call back functions
        self.contrast.callBackAutoPush = self.autoContrast
        self.contrast.callBackValueChange = self.changeContrast
        self.layout.addWidget(self.contrast.view)


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

    def setImage(self,image):
        print(" [VIEW] >> EditImageView.setImage(",image.name,")")
        return self.imageWidgetController.setImage(image)

    def autoExposure(self):
        print(" [CB] >> EditImageView.autoExposure(",")")

        self.controller.autoExposure()
        pass

    def changeExposure(self, value):
        print(" [CB] >> EditImageView.changeExposure(",")")

        self.controller.changeExposure(value)
        pass

    def autoContrast(self):
        print(" [CB] >> EditImageView.autoContrast(",")")
        pass

    def changeContrast(self, value):
        print(" [CB] >> EditImageView.changeContrast(",")")

        self.controller.changeContrast(value)
        pass

    def setProcessPipe(self, processPipe):
        print(" [VIEW] >> EditImageView.setProcessPipe(",")")
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
# ------------------------------------------------------------------------------------------
class MultiDockView(QDockWidget):
    def __init__(self, _controller):
        print(" [VIEW] >> MultiDockView.__init__(",")")

        super().__init__("Image Edit/Info")
        self.controller = _controller

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
      
        self.childControllers = [controller.EditImageController(self), controller.ImageInfoController(self)]
        self.childController = self.childControllers[0]
        self.active = 0
        self.setWidget(self.childController.view)
        self.repaint()

    def switch(self):
        print(" [VIEW] >> MultiDockView.switch(",")")
     
        self.active = (self.active+1)%len(self.childControllers)
        self.childController.view.deleteLater()
        self.childController = self.childControllers[self.active]
        # rebuild view 
        processPipe = self.controller.parent.imageGalleryController.getSelectedProcessPipe()
        self.childController.buildView(processPipe)
        self.setWidget(self.childController.view)
        self.repaint()

    def setProcessPipe(self, processPipe):
        print(" [VIEW] >> MultiDockView.setProcessPipe(",processPipe.getImage().name,")")
        self.childController.setProcessPipe(processPipe)
# ------------------------------------------------------------------------------------------
class AdvanceSliderView(QFrame):
    def __init__(self, controller, name,defaultValue, range, step):
        super().__init__()
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
        self.slider.setRange(range[0]/step,range[1]/step)
        self.slider.setValue(defaultValue/step)
        self.slider.setSingleStep(1)

        self.vbox.addWidget(self.firstrow)
        self.vbox.addWidget(self.slider)

        self.setLayout(self.vbox)

        # callBackFunctions slider/reset/auto
        self.slider.valueChanged.connect(self.controller.sliderChange)
        self.reset.clicked.connect(self.controller.reset)
        self.auto.clicked.connect(self.controller.auto)
# ------------------------------------------------------------------------------------------
class ImageUseCaseView(QSplitter):
    def __init__(self, _controller):
        print(" [VIEW] >> ImageUseCaseView.__init__(",")")

        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QFormLayout()

        # ---------------------------------------------------
        #1	Inside	Window with view on bright outdoor
        self.useCase01 = AdvanceRadioButton("[I] Window with view on bright outdoor", defaultState, layout, callBack=None)
        #2	Inside	High Contrast and Illuminants tend to be over-exposed
        self.useCase02 = AdvanceRadioButton("[I] High Contrast", False, self.layout, callBack=None)
        #3	Inside	Backlit portrait
        self.useCase03 = AdvanceRadioButton("[I] Backlit portrait", False, self.layout, callBack=None)
        #4	Outside	Sun in the frame
        self.useCase04 = AdvanceRadioButton("[I] Sun in the frame", False, self.layout, callBack=None)
        #5	Outside	Backlight
        self.useCase05 = AdvanceRadioButton("[O] Backlight", False, self.layout, callBack=None)
        #6	Outside	Shadow and direct lighting
        self.useCase06 = AdvanceRadioButton("[O] Shadow and direct lighting", False, self.layout, callBack=None)
        #7	Outside	Backlit portrait
        self.useCase07 = AdvanceRadioButton("[O] Backlit portrait", False, self.layout, callBack=None)
        #8	Outside	Nature
        self.useCase08 = AdvanceRadioButton("[O] Nature", False, self.layout, callBack=None)
        #9	Outside	Quite LDR
        self.useCase09 = AdvanceRadioButton("[O] Quite LDR", False, self.layout, callBack=None)
        #10	Lowlight	Portrait
        self.useCase10 = AdvanceRadioButton("[L] Portrait", False, self.layout, callBack=None)
        #11	Lowlight	Outside with bright illuminants
        self.useCase11 = AdvanceRadioButton("[L] Outside with bright illuminants", False, self.layout, callBack=None)
        #12	Lowlight	Cityscape
        self.useCase12 = AdvanceRadioButton("[L] Cityscape", False, self.layout, callBack=None)
        #13	Lowlight	Event (Concerts/night clubs/bar/restaurants)
        self.useCase13 = AdvanceRadioButton("[L] Event", False, self.layout, callBack=None)
        #14	Special cases	Shiny object  and Specular highlights
        self.useCase14 = AdvanceRadioButton("[S] Shiny object", False, self.layout, callBack=None)
        #15	Special cases	Memory colors
        self.useCase15 = AdvanceRadioButton("[S] Memory colors", False, self.layout, callBack=None)
        #16	Special cases	Scene with color checker / gray chart
        self.useCase16 = AdvanceRadioButton("[S] Color checker", False, self.layout, callBack=None)
        #17	Special cases	Translucent objects and Stained glass
        self.useCase17 = AdvanceRadioButton("[S] Translucent objects", False, self.layout, callBack=None)
        #18	Special cases	Traditional tone mapping failing cases
        self.useCase18 = AdvanceRadioButton("[S] TM failing cases", False, self.layout, callBack=None)
        # ---------------------------------------------------
        # ---------------------------------------------------
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        self.layout.addRow(line)
        # ---------------------------------------------------

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

    def setImage(self,image): 
        print(" [VIEW] >> ImageInfoView.setImage(",image.name,")")
        # ---------------------------------------------------
        return self.imageWidgetController.setImage(image)
# ------------------------------------------------------------------------------------------
class AdvanceRadioButton(object):
    ### https://www.tutorialspoint.com/pyqt/pyqt_qradiobutton_widget.htm
    def __init__(self, labelName, defaultState, layout, callBack=None):
        self.label = QLabel(labelName)
        self.radioButton =QRadioButton("test")
        if callBack: self.radioButton.toggled.connect(callBack)
        layout.addRow(self.label,self.radioButton)

    def setState(self, state): self.radioButton.setChecked(state)
# ------------------------------------------------------------------------------------------



