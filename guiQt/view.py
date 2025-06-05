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
    """A widget for displaying images."""

    def __init__(self, controller, colorData=None):
        """Initialize the ImageWidgetView.

        Args:
            controller: Reference to the controller
            colorData (numpy.ndarray, optional): Color data of the image. Defaults to None.
        """
        super().__init__()
        self.controller = controller
        self.label = QLabel(self)  # create a QtLabel for pixmap
        if not isinstance(colorData, np.ndarray):
            colorData = ImageWidgetView.emptyImageColorData()
        self.setPixmap(colorData)

    def resize(self):
        """Resize the widget and the label to fit the window size."""
        self.label.resize(self.size())
        self.label.setPixmap(self.imagePixmap.scaled(self.size(), Qt.KeepAspectRatio))

    def resizeEvent(self, event):
        """Handle the resize event.

        Args:
            event: The resize event
        """
        self.resize()
        super().resizeEvent(event)

    def setPixmap(self, colorData):
        """Set the pixmap from color data.

        Args:
            colorData (numpy.ndarray): Color data of the image

        Returns:
            QPixmap: The pixmap created from the color data
        """
        if not isinstance(colorData, np.ndarray):
            colorData = ImageWidgetView.emptyImageColorData()
        height, width, channel = colorData.shape  # compute pixmap
        bytesPerLine = channel * width
        # clip
        colorData[colorData > 1.0] = 1.0
        colorData[colorData < 0.0] = 0.0

        qImg = QImage((colorData * 255).astype(np.uint8), width, height, bytesPerLine, QImage.Format_RGB888)  # QImage
        self.imagePixmap = QPixmap.fromImage(qImg)
        self.resize()

        return self.imagePixmap

    def setQPixmap(self, qPixmap):
        """Set the pixmap directly from a QPixmap.

        Args:
            qPixmap (QPixmap): The QPixmap to set
        """
        self.imagePixmap = qPixmap
        self.resize()

    @staticmethod
    def emptyImageColorData():
        """Create an empty image color data array.

        Returns:
            numpy.ndarray: An array filled with a light gray color
        """
        return np.ones((90, 160, 3)) * (220 / 255)

class FigureWidget(FigureCanvas):
    """Matplotlib Figure Widget"""

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        """Initialize the FigureWidget.

        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
            width (int, optional): Width of the figure. Defaults to 5.
            height (int, optional): Height of the figure. Defaults to 5.
            dpi (int, optional): Dots per inch. Defaults to 100.
        """
        # create Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)  # explicit call of super constructor
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
        self.setMinimumSize(200, 200)

    def plot(self, X, Y, mode, clear=False):
        """Plot data on the figure.

        Args:
            X (array-like): X data to plot
            Y (array-like): Y data to plot
            mode (str): Plot mode
            clear (bool, optional): Whether to clear the axes before plotting. Defaults to False.
        """
        if clear:
            self.axes.clear()
        self.axes.plot(X, Y, mode)
        try:
            self.fig.canvas.draw()
        except Exception:
            time.sleep(0.5)
            self.fig.canvas.draw()

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

    def __init__(self, controller_=None, shapeMode=None):
        """Initialize the ImageGalleryView.

        Args:
            controller_ (optional): Reference to the controller. Defaults to None.
            shapeMode (optional): Shape mode for the gallery. Defaults to None.
        """
        if pref.verbose:
            print(" [VIEW] >> ImageGalleryView.__init__()")

        super().__init__(Qt.Vertical)

        self.controller = controller_
        self.shapeMode = controller.GalleryMode._3x2 if not shapeMode else shapeMode  # default display mode
        self.pageNumber = 0

        self.imagesControllers = []

        self.images = QFrame()
        self.images.setFrameShape(QFrame.StyledPanel)
        self.imagesLayout = QGridLayout()
        self.images.setLayout(self.imagesLayout)

        self.buildGridLayoutWidgets()

        self.previousPageButton = QPushButton('<')
        self.previousPageButton.clicked.connect(self.controller.callBackButton_previousPage)
        self._1x1Button = QPushButton('1x1')
        self._1x1Button.clicked.connect(self.controller.callBackButton_1x1)
        self._2x1Button = QPushButton('2x1')
        self._2x1Button.clicked.connect(self.controller.callBackButton_2x1)
        self._3x2Button = QPushButton('3x2')
        self._3x2Button.clicked.connect(self.controller.callBackButton_3x2)
        self._6x4Button = QPushButton('6x4')
        self._6x4Button.clicked.connect(self.controller.callBackButton_6x4)
        self._9x6Button = QPushButton('9x6')
        self._9x6Button.clicked.connect(self.controller.callBackButton_9x6)
        self.nextPageButton = QPushButton('>')
        self.nextPageButton.clicked.connect(self.controller.callBackButton_nextPage)

        self.pageNumberLabel = QLabel(str(self.pageNumber) + "/ ...")

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
        self.setSizes([1525, 82])

    def currentPage(self):
        """Get the current page number.

        Returns:
            int: Current page number
        """
        return self.pageNumber

    def changePageNumber(self, step):
        """Change the current page number by a given step.

        Args:
            step (int): Step to change the page number
        """
        if pref.verbose:
            print(" [VIEW] >> ImageGalleryView.changePageNumber(", step, ")")

        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode) * controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes) - 1) // nbImagePerPage) + 1

        if len(self.controller.model.processPipes) > 0:
            oldPageNumber = self.pageNumber
            if (self.pageNumber + step) > maxPage - 1:
                self.pageNumber = 0
            elif (self.pageNumber + step) < 0:
                self.pageNumber = maxPage - 1
            else:
                self.pageNumber = self.pageNumber + step
            self.updateImages()
            self.controller.model.loadPage(self.pageNumber)
            if pref.verbose:
                print(" [VIEW] >> ImageGalleryView.changePageNumber(currentPage:", self.pageNumber, "| max page:", maxPage, ")")

    def updateImages(self):
        """Update the images in the gallery."""
        if pref.verbose:
            print(" [VIEW] >> ImageGalleryView.updateImages()")

        nbImagePerPage = controller.GalleryMode.nbRow(self.shapeMode) * controller.GalleryMode.nbCol(self.shapeMode)
        maxPage = ((len(self.controller.model.processPipes) - 1) // nbImagePerPage) + 1

        index = 0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)):
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                iwc = self.imagesControllers[index]
                iwc.view.setPixmap(ImageWidgetView.emptyImageColorData())
                index += 1
        self.pageNumberLabel.setText(str(self.pageNumber) + "/" + str(maxPage - 1))

    def updateImage(self, idx, processPipe, filename):
        """Update a specific image in the gallery.

        Args:
            idx (int): Index of the image to update
            processPipe: Process pipe associated with the image
            filename (str): Filename of the image
        """
        if pref.verbose:
            print(" [VIEW] >> ImageGalleryView.updateImage()")
        imageWidgetController = self.imagesControllers[idx]
        imageWidgetController.setImage(processPipe.getImage())
        self.controller.parent.statusBar().showMessage("loading of image " + filename + " done!")

    def resetGridLayoutWidgets(self):
        """Reset the grid layout widgets."""
        if pref.verbose:
            print(" [VIEW] >> ImageGalleryView.resetGridLayoutWidgets()")

        for w in self.imagesControllers:
            self.imagesLayout.removeWidget(w.view)
            w.view.deleteLater()
        self.imagesControllers = []

    def buildGridLayoutWidgets(self):
        """Build the grid layout widgets."""
        if pref.verbose:
            print(" [VIEW] >> ImageGalleryView.buildGridLayoutWidgets()")

        imageIndex = 0
        for i in range(controller.GalleryMode.nbRow(self.shapeMode)):
            for j in range(controller.GalleryMode.nbCol(self.shapeMode)):
                iwc = controller.ImageWidgetController(id=imageIndex)
                self.imagesControllers.append(iwc)
                self.imagesLayout.addWidget(iwc.view, i, j)
                imageIndex += 1

    def wheelEvent(self, event):
        """Handle the wheel event for changing pages.

        Args:
            event: The wheel event
        """
        if pref.verbose:
            print(" [EVENT] >> ImageGalleryView.wheelEvent()")

        if event.angleDelta().y() < 0:
            self.changePageNumber(+1)
            if self.shapeMode == controller.GalleryMode._1x1:
                self.controller.selectImage(0)

        else:
            self.changePageNumber(-1)
            if self.shapeMode == controller.GalleryMode._1x1:
                self.controller.selectImage(0)
        event.accept()

    def mousePressEvent(self, event):
        """Handle the mouse press event for selecting images.

        Args:
            event: The mouse press event
        """
        if pref.verbose:
            print(" [EVENT] >> ImageGalleryView.mousePressEvent()")

        if isinstance(self.childAt(event.pos()).parent(), ImageWidgetView):
            id = self.childAt(event.pos()).parent().controller.id()
        else:
            id = -1

        if id != -1:  # an image is clicked, select it!
            self.controller.selectImage(id)
        event.accept()

class AppView(QMainWindow):
    """
    MainWindow(View)
    """

    def __init__(self, _controller=None, shapeMode=None, HDRcontroller=None):
        """Initialize the AppView.

        Args:
            _controller (optional): Reference to the controller. Defaults to None.
            shapeMode (optional): Shape mode for the gallery. Defaults to None.
            HDRcontroller (optional): Reference to the HDR controller. Defaults to None.
        """
        super().__init__()
        scale = 0.8
        self.controller = _controller
        self.setWindowGeometry(scale=scale)
        self.setWindowTitle('uHDR - Rémi Cozot (c) 2020-2021')  # title
        self.statusBar().showMessage('Welcome to uHDR!')  # status bar

        self.topContainer = QWidget()
        self.topLayout = QHBoxLayout()

        self.imageGalleryController = controller.ImageGalleryController(self)

        self.topLayout.addWidget(self.imageGalleryController.view)

        self.topContainer.setLayout(self.topLayout)
        self.setCentralWidget(self.topContainer)
        self.dock = controller.MultiDockController(self, HDRcontroller)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock.view)
        self.resizeDocks([self.dock.view], [int(self.controller.screenSize[0].width() * scale // 4)], Qt.Horizontal)

        self.buildFileMenu()
        self.buildDockMenu()
        self.buildDisplayHDR()
        self.buildExport()
        self.buildPreferences()

    def getImageGalleryController(self):
        """Get the image gallery controller.

        Returns:
            ImageGalleryController: The image gallery controller
        """
        return self.imageGalleryController

    def resizeEvent(self, event):
        """Handle the resize event.

        Args:
            event: The resize event
        """
        super().resizeEvent(event)

    def setWindowGeometry(self, scale=0.8):
        """Set the window geometry.

        Args:
            scale (float, optional): Scale factor for the window size. Defaults to 0.8.
        """
        displayCoord = QDesktopWidget().screenGeometry(1)
        if len(self.controller.screenSize) > 1:
            width, height = self.controller.screenSize[1].width(), self.controller.screenSize[1].height()
        else:
            width, height = self.controller.screenSize[0].width(), self.controller.screenSize[0].height()

        self.setGeometry(displayCoord.left(), displayCoord.top() + 50, math.floor(width * scale), math.floor(height * scale))
        self.showMaximized()

    def buildFileMenu(self):
        """Build the file menu."""
        menubar = self.menuBar()  # get menubar
        fileMenu = menubar.addMenu('&File')  # file menu

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

    def buildPreferences(self):
        """Build the preferences menu."""
        menubar = self.menuBar()  # get menubar
        prefMenu = menubar.addMenu('&Preferences')  # file menu

        displayList = pref.getHDRdisplays().keys()

        def cbd(tag):
            """Callback function for setting the display.

            Args:
                tag: Display tag
            """
            pref.setHDRdisplay(tag)
            self.statusBar().showMessage("switching HDR Display to: " + tag + "!")
            self.menuExport.setText('&Export to ' + pref.getHDRdisplay()['tag'])
            self.menuExportAll.setText('&Export All to ' + pref.getHDRdisplay()['tag'])

        prefDisplays = []
        for i, d in enumerate(displayList):
            if d != 'none':
                prefDisplay = QAction('&Set display to ' + d, self)
                p_cbd = functools.partial(cbd, d)
                prefDisplay.triggered.connect(p_cbd)
                prefMenu.addAction(prefDisplay)

    def buildDisplayHDR(self):
        """Build the display HDR menu."""
        menubar = self.menuBar()  # get menubar
        displayHDRmenu = menubar.addMenu('&Display HDR')  # file menu

        displayHDR = QAction('&Display HDR image', self)
        displayHDR.setShortcut('Ctrl+H')
        displayHDR.setStatusTip('[Display HDR] display HDR image')
        displayHDR.triggered.connect(self.controller.callBackDisplayHDR)
        displayHDRmenu.addAction(displayHDR)

        displayHDR = QAction('&Compare raw and edited HDR image', self)
        displayHDR.setShortcut('Ctrl+C')
        displayHDR.setStatusTip('[Display HDR] compare raw HDR image and edited one')
        displayHDR.triggered.connect(self.controller.callBackCompareRawEditedHDR)
        displayHDRmenu.addAction(displayHDR)

        closeHDR = QAction('&reset HDR display', self)
        closeHDR.setShortcut('Ctrl+K')
        closeHDR.setStatusTip('[Display HDR] reset HDR window')
        closeHDR.triggered.connect(self.controller.callBackCloseDisplayHDR)
        displayHDRmenu.addAction(closeHDR)

    def buildExport(self):
        """Build the export menu."""
        menubar = self.menuBar()  # get menubar
        exportHDR = menubar.addMenu('&Export HDR image')  # file menu

        self.menuExport = QAction('&Export to ' + pref.getHDRdisplay()['tag'], self)
        self.menuExport.setShortcut('Ctrl+X')
        self.menuExport.setStatusTip('[Export HDR image] save HDR image file for HDR display')
        self.menuExport.triggered.connect(self.controller.callBackExportHDR)
        exportHDR.addAction(self.menuExport)

        self.menuExportAll = QAction('&Export All to ' + pref.getHDRdisplay()['tag'], self)
        self.menuExportAll.setShortcut('Ctrl+Y')
        self.menuExportAll.setStatusTip('[Export all HDR images] save HDR image files for HDR display.')
        self.menuExportAll.triggered.connect(self.controller.callBackExportAllHDR)
        exportHDR.addAction(self.menuExportAll)

    def buildDockMenu(self):
        """Build the dock menu."""
        menubar = self.menuBar()  # get menubar
        dockMenu = menubar.addMenu('&Dock')  # file menu

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

    def closeEvent(self, event):
        """Handle the close event.

        Args:
            event: The close event
        """
        if pref.verbose:
            print(" [CB] >> AppView.closeEvent()>> ... closing")
        self.imageGalleryController.save()
        self.controller.hdrDisplay.close()

# ------------------------------------------------------------------------------------------
# --- class ImageInfoView(QSplitter) -------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ImageInfoView(QSplitter):
    """A view for displaying image information and metadata."""

    def __init__(self, _controller):
        """Initialize the ImageInfoView.

        Args:
            _controller: Reference to the controller
        """
        if pref.verbose:
            print(" [VIEW] >> ImageInfoView.__init__()")

        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QFormLayout()

        # Image information fields
        self.imageName = AdvanceLineEdit(" name:", " ........ ", self.layout, callBack=None)
        self.imagePath = AdvanceLineEdit(" path:", " ........ ", self.layout, callBack=None)
        self.imageSize = AdvanceLineEdit(" size (pixel):", ".... x .... ", self.layout, callBack=None)
        self.imageDynamicRange = AdvanceLineEdit(" dynamic range (f-stops)", " ........ ", self.layout, callBack=None)
        self.colorSpace = AdvanceLineEdit(" color space:", " ........ ", self.layout, callBack=None)
        self.imageType = AdvanceLineEdit(" type:", " ........ ", self.layout, callBack=None)
        self.imageBPS = AdvanceLineEdit(" bits per sample:", " ........ ", self.layout, callBack=None)
        self.imageExpoTime = AdvanceLineEdit(" exposure time:", " ........ ", self.layout, callBack=None)
        self.imageFNumber = AdvanceLineEdit("f-number:", " ........ ", self.layout, callBack=None)
        self.imageISO = AdvanceLineEdit(" ISO:", " ........ ", self.layout, callBack=None)
        self.imageCamera = AdvanceLineEdit(" camera:", " ........ ", self.layout, callBack=None)
        self.imageSoftware = AdvanceLineEdit(" software:", " ........ ", self.layout, callBack=None)
        self.imageLens = AdvanceLineEdit(" lens:", " ........ ", self.layout, callBack=None)
        self.imageFocalLength = AdvanceLineEdit(" focal length:", " ........ ", self.layout, callBack=None)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        self.layout.addRow(line)

        # User-defined tags
        userDefinedTags = hdrCore.metadata.tags()
        tagRootName = userDefinedTags.getTagsRootName()
        listOfTags = userDefinedTags.tags[tagRootName]
        self.userDefinedTags = []
        for tagGroup in listOfTags:
            groupKey = list(tagGroup.keys())[0]
            tagLeafs = tagGroup[groupKey]
            for tag in tagLeafs.items():
                self.userDefinedTags.append(AdvanceCheckBox(self, groupKey, tag[0], False, self.layout))
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            self.layout.addRow(line)

        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60, 40])

    def setProcessPipe(self, processPipe):
        """Set the process pipe and update the view with image metadata.

        Args:
            processPipe: The process pipe to set
        """
        image_ = processPipe.getImage()
        if pref.verbose:
            print(" [VIEW] >> ImageInfoView.setImage(", image_.name, ")")
        if image_.metadata.metadata['filename'] is not None:
            self.imageName.setText(image_.metadata.metadata['filename'])
        else:
            self.imageName.setText(" ........ ")
        if image_.metadata.metadata['path'] is not None:
            self.imagePath.setText(image_.metadata.metadata['path'])
        else:
            self.imagePath.setText(" ........ ")
        if image_.metadata.metadata['exif']['Image Width'] is not None:
            self.imageSize.setText(str(image_.metadata.metadata['exif']['Image Width']) + " x " + str(image_.metadata.metadata['exif']['Image Height']))
        else:
            self.imageSize.setText(" ........ ")
        if image_.metadata.metadata['exif']['Dynamic Range (stops)'] is not None:
            self.imageDynamicRange.setText(str(image_.metadata.metadata['exif']['Dynamic Range (stops)']))
        else:
            self.imageDynamicRange.setText(" ........ ")
        if image_.metadata.metadata['exif']['Color Space'] is not None:
            self.colorSpace.setText(image_.metadata.metadata['exif']['Color Space'])
        else:
            self.imageName.setText(" ........ ")
        if image_.type is not None:
            self.imageType.setText(str(image_.type))
        else:
            self.colorSpace.setText(" ........ ")
        if image_.metadata.metadata['exif']['Bits Per Sample'] is not None:
            self.imageBPS.setText(str(image_.metadata.metadata['exif']['Bits Per Sample']))
        else:
            self.imageBPS.setText(" ........ ")
        if image_.metadata.metadata['exif']['Exposure Time'] is not None:
            self.imageExpoTime.setText(str(image_.metadata.metadata['exif']['Exposure Time'][0]) + " / " + str(image_.metadata.metadata['exif']['Exposure Time'][1]))
        else:
            self.imageExpoTime.setText(" ........ ")
        if image_.metadata.metadata['exif']['F Number'] is not None:
            self.imageFNumber.setText(str(image_.metadata.metadata['exif']['F Number'][0]))
        else:
            self.imageFNumber.setText(" ........ ")
        if image_.metadata.metadata['exif']['ISO'] is not None:
            self.imageISO.setText(str(image_.metadata.metadata['exif']['ISO']))
        else:
            self.imageISO.setText(" ........ ")
        if image_.metadata.metadata['exif']['Camera'] is not None:
            self.imageCamera.setText(image_.metadata.metadata['exif']['Camera'])
        else:
            self.imageCamera.setText(" ........ ")
        if image_.metadata.metadata['exif']['Software'] is not None:
            self.imageSoftware.setText(image_.metadata.metadata['exif']['Software'])
        else:
            self.imageSoftware.setText(" ........ ")
        if image_.metadata.metadata['exif']['Lens'] is not None:
            self.imageLens.setText(image_.metadata.metadata['exif']['Lens'])
        else:
            self.imageLens.setText(" ........ ")
        if image_.metadata.metadata['exif']['Focal Length'] is not None:
            self.imageFocalLength.setText(str(image_.metadata.metadata['exif']['Focal Length'][0]))
        else:
            self.imageFocalLength.setText(" ........ ")

        self.controller.callBackActive = False

        tagRootName = image_.metadata.otherTags.getTagsRootName()
        listOfTags = image_.metadata.metadata[tagRootName]

        for i, tagGroup in enumerate(listOfTags):
            groupKey = list(tagGroup.keys())[0]
            tagLeafs = tagGroup[groupKey]
            for tag in tagLeafs.items():
                for acb in self.userDefinedTags:
                    if (acb.rightText == tag[0]) and (acb.leftText == groupKey):
                        on_off = image_.metadata.metadata[tagRootName][i][groupKey][tag[0]]
                        on_off = on_off if on_off else False
                        acb.setState(on_off)
                        break

        self.controller.callBackActive = True

        return self.imageWidgetController.setImage(image_)

    def metadataChange(self, metaGroup, metaTag, on_off):
        """Handle metadata changes.

        Args:
            metaGroup: The group of the metadata
            metaTag: The tag of the metadata
            on_off: The new value of the metadata
        """
        if self.controller.callBackActive:
            self.controller.metadataChange(metaGroup, metaTag, on_off)

class AdvanceLineEdit(object):
    """A custom QLineEdit widget with a label."""

    def __init__(self, labelName, defaultText, layout, callBack=None):
        """Initialize the AdvanceLineEdit.

        Args:
            labelName (str): The name of the label
            defaultText (str): The default text for the line edit
            layout: The layout to add the widget to
            callBack (optional): Callback function for text changes. Defaults to None.
        """
        self.label = QLabel(labelName)
        self.lineEdit = QLineEdit(defaultText)
        if callBack:
            self.lineEdit.textChanged.connect(callBack)
        layout.addRow(self.label, self.lineEdit)

    def setText(self, txt):
        """Set the text of the line edit.

        Args:
            txt (str): The text to set
        """
        self.lineEdit.setText(txt)

class AdvanceCheckBox(object):
    """A custom QCheckBox widget with a label."""

    def __init__(self, parent, leftText, rightText, defaultValue, layout):
        """Initialize the AdvanceCheckBox.

        Args:
            parent: Reference to the parent widget
            leftText (str): The text for the label
            rightText (str): The text for the checkbox
            defaultValue: The default value of the checkbox
            layout: The layout to add the widget to
        """
        self.parent = parent
        self.leftText = leftText
        self.rightText = rightText

        self.label = QLabel(leftText)
        self.checkbox = QCheckBox(rightText)
        self.checkbox.toggled.connect(self.toggled)
        layout.addRow(self.label, self.checkbox)

    def setState(self, on_off):
        """Set the state of the checkbox.

        Args:
            on_off (bool): The state to set
        """
        self.checkbox.setChecked(on_off)

    def toggled(self):
        """Handle the toggled event of the checkbox."""
        self.parent.metadataChange(self.leftText, self.rightText, self.checkbox.isChecked())

class EditImageView(QSplitter):
    """A view for editing images with various controls."""

    def __init__(self, _controller, build=False):
        """Initialize the EditImageView.

        Args:
            _controller: Reference to the controller
            build (bool, optional): Whether to build the view. Defaults to False.
        """
        if pref.verbose:
            print(" [VIEW] >> EditImageView.__init__()")
        super().__init__(Qt.Vertical)

        self.controller = _controller

        self.imageWidgetController = controller.ImageWidgetController()

        self.layout = QVBoxLayout()

        # Exposure control
        self.exposure = controller.AdvanceSliderController(self, "exposure", 0, (-10, +10), 0.25)
        self.exposure.callBackAutoPush = self.autoExposure
        self.exposure.callBackValueChange = self.changeExposure
        self.layout.addWidget(self.exposure.view)

        # Contrast control
        self.contrast = controller.AdvanceSliderController(self, "contrast", 0, (-100, +100), 1)
        self.contrast.callBackAutoPush = self.autoContrast
        self.contrast.callBackValueChange = self.changeContrast
        self.layout.addWidget(self.contrast.view)

        # Tone curve control
        self.tonecurve = controller.ToneCurveController(self)
        self.layout.addWidget(self.tonecurve.view)

        # Lightness mask control
        self.lightnessmask = controller.LightnessMaskController(self)
        self.layout.addWidget(self.lightnessmask.view)

        # Saturation control
        self.saturation = controller.AdvanceSliderController(self, "saturation", 0, (-100, +100), 1)
        self.saturation.callBackAutoPush = self.autoSaturation
        self.saturation.callBackValueChange = self.changeSaturation
        self.layout.addWidget(self.saturation.view)

        # Color editor controls
        self.colorEditor0 = controller.LchColorSelectorController(self, idName="colorEditor0")
        self.layout.addWidget(self.colorEditor0.view)

        self.colorEditor1 = controller.LchColorSelectorController(self, idName="colorEditor1")
        self.layout.addWidget(self.colorEditor1.view)

        self.colorEditor2 = controller.LchColorSelectorController(self, idName="colorEditor2")
        self.layout.addWidget(self.colorEditor2.view)

        self.colorEditor3 = controller.LchColorSelectorController(self, idName="colorEditor3")
        self.layout.addWidget(self.colorEditor3.view)

        self.colorEditor4 = controller.LchColorSelectorController(self, idName="colorEditor4")
        self.layout.addWidget(self.colorEditor4.view)

        # Auto color selection control
        self.colorEditorsAuto = controller.ColorEditorsAutoController(self,
                                                                      [self.colorEditor0,
                                                                       self.colorEditor1,
                                                                       self.colorEditor2,
                                                                       self.colorEditor3,
                                                                       self.colorEditor4],
                                                                      "saturation")
        self.layout.addWidget(self.colorEditorsAuto.view)

        # Geometry control
        self.geometry = controller.GeometryController(self)
        self.layout.addWidget(self.geometry.view)

        # HDR preview control
        self.hdrPreview = HDRviewerView(self.controller.controllerHDR, build)
        self.controller.controllerHDR.setView(self.hdrPreview)
        self.layout.addWidget(self.hdrPreview)

        # Scroll area
        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        # Adding widgets to self (QSplitter)
        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60, 40])

    def setImage(self, image):
        """Set the image to be displayed.

        Args:
            image: The image to set

        Returns:
            The result of setting the image in the image widget controller
        """
        if pref.verbose:
            print(" [VIEW] >> EditImageView.setImage(", image.name, ")")
        return self.imageWidgetController.setImage(image)

    def autoExposure(self):
        """Handle the auto exposure button click."""
        if pref.verbose:
            print(" [CB] >> EditImageView.autoExposure()")
        self.controller.autoExposure()

    def changeExposure(self, value):
        """Handle the exposure value change.

        Args:
            value: The new exposure value
        """
        if pref.verbose:
            print(" [CB] >> EditImageView.changeExposure()")
        self.controller.changeExposure(value)

    def autoContrast(self):
        """Handle the auto contrast button click."""
        if pref.verbose:
            print(" [CB] >> EditImageView.autoContrast()")

    def changeContrast(self, value):
        """Handle the contrast value change.

        Args:
            value: The new contrast value
        """
        if pref.verbose:
            print(" [CB] >> EditImageView.changeContrast()")
        self.controller.changeContrast(value)

    def autoSaturation(self):
        """Handle the auto saturation button click."""
        if pref.verbose:
            print(" [CB] >> EditImageView.autoSaturation()")

    def changeSaturation(self, value):
        """Handle the saturation value change.

        Args:
            value: The new saturation value
        """
        if pref.verbose:
            print(" [CB] >> EditImageView.changeSaturation()")
        self.controller.changeSaturation(value)

    def plotToneCurve(self):
        """Plot the tone curve."""
        self.tonecurve.plotCurve()

    def setProcessPipe(self, processPipe):
        """Set the process pipe and initialize the view components with its parameters.

        Args:
            processPipe: The process pipe to set
        """
        if pref.verbose:
            print(" [VIEW] >> EditImageView.setProcessPipe()")

        # Exposure
        id = processPipe.getProcessNodeByName("exposure")
        value = processPipe.getParameters(id)
        self.exposure.setValue(value['EV'], callBackActive=False)

        # Contrast
        id = processPipe.getProcessNodeByName("contrast")
        value = processPipe.getParameters(id)
        self.contrast.setValue(value['contrast'], callBackActive=False)

        # Tone curve
        id = processPipe.getProcessNodeByName("tonecurve")
        value = processPipe.getParameters(id)
        self.tonecurve.setValues(value, callBackActive=False)

        # Lightness mask
        id = processPipe.getProcessNodeByName("lightnessmask")
        value = processPipe.getParameters(id)
        self.lightnessmask.setValues(value, callBackActive=False)

        # Saturation
        id = processPipe.getProcessNodeByName("saturation")
        value = processPipe.getParameters(id)
        self.saturation.setValue(value['saturation'], callBackActive=False)

        # Color Editor 0
        id = processPipe.getProcessNodeByName("colorEditor0")
        values = processPipe.getParameters(id)
        self.colorEditor0.setValues(values, callBackActive=False)

        # Color Editor 1
        id = processPipe.getProcessNodeByName("colorEditor1")
        values = processPipe.getParameters(id)
        self.colorEditor1.setValues(values, callBackActive=False)

        # Color Editor 2
        id = processPipe.getProcessNodeByName("colorEditor2")
        values = processPipe.getParameters(id)
        self.colorEditor2.setValues(values, callBackActive=False)

        # Color Editor 3
        id = processPipe.getProcessNodeByName("colorEditor3")
        values = processPipe.getParameters(id)
        self.colorEditor3.setValues(values, callBackActive=False)

        # Color Editor 4
        id = processPipe.getProcessNodeByName("colorEditor4")
        values = processPipe.getParameters(id)
        self.colorEditor4.setValues(values, callBackActive=False)

        # Geometry
        id = processPipe.getProcessNodeByName("geometry")
        values = processPipe.getParameters(id)
        self.geometry.setValues(values, callBackActive=False)

class MultiDockView(QDockWidget):
    """A dock widget for switching between different views."""

    def __init__(self, _controller, HDRcontroller=None):
        """Initialize the MultiDockView.

        Args:
            _controller: Reference to the controller
            HDRcontroller (optional): Reference to the HDR controller. Defaults to None.
        """
        if pref.verbose:
            print(" [VIEW] >> MultiDockView.__init__()")

        super().__init__("Image Edit/Info")
        self.controller = _controller

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.childControllers = [
            controller.EditImageController(self, HDRcontroller),
            controller.ImageInfoController(self),
            controller.ImageAestheticsController(self)
        ]
        self.childController = self.childControllers[0]
        self.active = 0
        self.setWidget(self.childController.view)
        self.repaint()

    def switch(self, nb):
        """Switch the active dock view.

        Args:
            nb (int): The index of the view to switch to
        """
        if pref.verbose:
            print(" [VIEW] >> MultiDockView.switch(", nb, ")")

        if nb != self.active:
            self.active = (nb) % len(self.childControllers)
            self.childController.view.deleteLater()
            self.childController = self.childControllers[self.active]
            processPipe = self.controller.parent.imageGalleryController.getSelectedProcessPipe()
            self.childController.buildView(processPipe)
            self.setWidget(self.childController.view)
            self.repaint()

    def setProcessPipe(self, processPipe):
        """Set the process pipe for the active child controller.

        Args:
            processPipe: The process pipe to set

        Returns:
            The result of setting the process pipe in the child controller
        """
        if pref.verbose:
            print(" [VIEW] >> MultiDockView.setProcessPipe(", processPipe.getImage().name, ")")
        return self.childController.setProcessPipe(processPipe)

class AdvanceSliderView(QFrame):
    """A custom slider widget with additional controls."""

    def __init__(self, controller, name, defaultValue, range, step):
        """Initialize the AdvanceSliderView.

        Args:
            controller: Reference to the controller
            name (str): The name of the slider
            defaultValue: The default value of the slider
            range (tuple): The range of the slider
            step (float): The step size of the slider
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = controller
        self.firstrow = QFrame()

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()

        self.firstrow.setLayout(self.hbox)

        self.label = QLabel(name)
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
        self.slider.setRange(int(range[0] / step), int(range[1] / step))
        self.slider.setValue(int(defaultValue / step))
        self.slider.setSingleStep(1)

        self.vbox.addWidget(self.firstrow)
        self.vbox.addWidget(self.slider)

        self.setLayout(self.vbox)

        # Callback functions for slider, reset, and auto buttons
        self.slider.valueChanged.connect(self.controller.sliderChange)
        self.reset.clicked.connect(self.controller.reset)
        self.auto.clicked.connect(self.controller.auto)

# ------------------------------------------------------------------------------------------
# --- class ToneCurveView(QFrame) ----------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ToneCurveView(QFrame):
    """A view for adjusting the tone curve of an image."""

    def __init__(self, controller):
        """Initialize the ToneCurveView.

        Args:
            controller: Reference to the controller
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = controller
        self.vbox = QVBoxLayout()

        # Figure
        self.curve = FigureWidget(self)
        self.curve.setMinimumSize(200, 600)
        self.curve.plot([0.0, 100], [0.0, 100.0], 'r--')

        # Containers
        self.containerAuto = QFrame()
        self.hboxAuto = QHBoxLayout()
        self.containerAuto.setLayout(self.hboxAuto)

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
        self.vbox.addWidget(self.containerAuto)
        self.vbox.addWidget(self.containerHighlights)
        self.vbox.addWidget(self.containerWhites)
        self.vbox.addWidget(self.containerMediums)
        self.vbox.addWidget(self.containerBlacks)
        self.vbox.addWidget(self.containerShadows)

        # Auto curve button
        self.autoCurve = QPushButton("auto")
        self.hboxAuto.addWidget(self.autoCurve)
        self.hboxAuto.setAlignment(Qt.AlignCenter)

        # Shadows
        self.labelShadows = QLabel("shadows")
        self.sliderShadows = QSlider(Qt.Horizontal)
        self.sliderShadows.setRange(0, 10000)
        self.sliderShadows.setValue(int(self.controller.model.default['shadows'][1]) * 100)
        self.sliderShadows.setTickInterval(25)
        self.sliderShadows.setSingleStep(25)
        self.sliderShadows.setPageStep(100)
        self.editShadows = QLineEdit()
        self.editShadows.setText(str(self.controller.model.default['shadows'][1]))
        self.resetShadows = QPushButton("reset")
        self.hboxShadows.addWidget(self.labelShadows)
        self.hboxShadows.addWidget(self.sliderShadows)
        self.hboxShadows.addWidget(self.editShadows)
        self.hboxShadows.addWidget(self.resetShadows)

        # Blacks
        self.labelBlacks = QLabel("blacks")
        self.sliderBlacks = QSlider(Qt.Horizontal)
        self.sliderBlacks.setRange(0, 10000)
        self.sliderBlacks.setValue(int(self.controller.model.default['blacks'][1]) * 100)
        self.editBlacks = QLineEdit()
        self.editBlacks.setText(str(self.controller.model.default['blacks'][1]))
        self.resetBlacks = QPushButton("reset")
        self.hboxBlacks.addWidget(self.labelBlacks)
        self.hboxBlacks.addWidget(self.sliderBlacks)
        self.hboxBlacks.addWidget(self.editBlacks)
        self.hboxBlacks.addWidget(self.resetBlacks)

        # Mediums
        self.labelMediums = QLabel("mediums")
        self.sliderMediums = QSlider(Qt.Horizontal)
        self.sliderMediums.setRange(0, 10000)
        self.sliderMediums.setValue(int(self.controller.model.default['mediums'][1]) * 100)
        self.editMediums = QLineEdit()
        self.editMediums.setText(str(self.controller.model.default['mediums'][1]))
        self.resetMediums = QPushButton("reset")
        self.hboxMediums.addWidget(self.labelMediums)
        self.hboxMediums.addWidget(self.sliderMediums)
        self.hboxMediums.addWidget(self.editMediums)
        self.hboxMediums.addWidget(self.resetMediums)

        # Whites
        self.labelWhites = QLabel("whites")
        self.sliderWhites = QSlider(Qt.Horizontal)
        self.sliderWhites.setRange(0, 10000)
        self.sliderWhites.setValue(int(self.controller.model.default['whites'][1]) * 100)
        self.editWhites = QLineEdit()
        self.editWhites.setText(str(self.controller.model.default['whites'][1]))
        self.resetWhites = QPushButton("reset")
        self.hboxWhites.addWidget(self.labelWhites)
        self.hboxWhites.addWidget(self.sliderWhites)
        self.hboxWhites.addWidget(self.editWhites)
        self.hboxWhites.addWidget(self.resetWhites)

        # Highlights
        self.labelHighlights = QLabel("highlights")
        self.sliderHighlights = QSlider(Qt.Horizontal)
        self.sliderHighlights.setRange(0, 10000)
        self.sliderHighlights.setValue(int(self.controller.model.default['highlights'][1]) * 100)
        self.editHighlights = QLineEdit()
        self.editHighlights.setText(str(self.controller.model.default['highlights'][1]))
        self.resetHighlights = QPushButton("reset")
        self.hboxHighlights.addWidget(self.labelHighlights)
        self.hboxHighlights.addWidget(self.sliderHighlights)
        self.hboxHighlights.addWidget(self.editHighlights)
        self.hboxHighlights.addWidget(self.resetHighlights)

        self.setLayout(self.vbox)

        # Callback functions for sliders and reset buttons
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
        self.autoCurve.clicked.connect(self.controller.autoCurve)

    def sliderShadowsChange(self):
        """Handle the change of the shadows slider."""
        if self.controller.callBackActive:
            value = self.sliderShadows.value() / 100
            self.controller.sliderChange("shadows", value)

    def sliderBlacksChange(self):
        """Handle the change of the blacks slider."""
        if self.controller.callBackActive:
            value = self.sliderBlacks.value() / 100
            self.controller.sliderChange("blacks", value)

    def sliderMediumsChange(self):
        """Handle the change of the mediums slider."""
        if self.controller.callBackActive:
            value = self.sliderMediums.value() / 100
            self.controller.sliderChange("mediums", value)

    def sliderWhitesChange(self):
        """Handle the change of the whites slider."""
        if self.controller.callBackActive:
            value = self.sliderWhites.value() / 100
            self.controller.sliderChange("whites", value)

    def sliderHighlightsChange(self):
        """Handle the change of the highlights slider."""
        if self.controller.callBackActive:
            value = self.sliderHighlights.value() / 100
            self.controller.sliderChange("highlights", value)

    def resetShadowsCB(self):
        """Reset the shadows slider to its default value."""
        if self.controller.callBackActive:
            self.controller.reset("shadows")

    def resetBlacksCB(self):
        """Reset the blacks slider to its default value."""
        if self.controller.callBackActive:
            self.controller.reset("blacks")

    def resetMediumsCB(self):
        """Reset the mediums slider to its default value."""
        if self.controller.callBackActive:
            self.controller.reset("mediums")

    def resetWhitesCB(self):
        """Reset the whites slider to its default value."""
        if self.controller.callBackActive:
            self.controller.reset("whites")

    def resetHighlightsCB(self):
        """Reset the highlights slider to its default value."""
        if self.controller.callBackActive:
            self.controller.reset("highlights")

class LightnessMaskView(QGroupBox):
    """A view for adjusting the lightness mask of an image."""

    def __init__(self, _controller):
        """Initialize the LightnessMaskView.

        Args:
            _controller: Reference to the controller
        """
        super().__init__("mask lightness")
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

    def clickShadows(self):
        """Handle the toggling of the shadows checkbox."""
        if self.controller.callBackActive:
            self.controller.maskChange("shadows", self.checkboxShadows.isChecked())

    def clickBlacks(self):
        """Handle the toggling of the blacks checkbox."""
        if self.controller.callBackActive:
            self.controller.maskChange("blacks", self.checkboxBlacks.isChecked())

    def clickMediums(self):
        """Handle the toggling of the mediums checkbox."""
        if self.controller.callBackActive:
            self.controller.maskChange("mediums", self.checkboxMediums.isChecked())

    def clickWhites(self):
        """Handle the toggling of the whites checkbox."""
        if self.controller.callBackActive:
            self.controller.maskChange("whites", self.checkboxWhites.isChecked())

    def clickHighlights(self):
        """Handle the toggling of the highlights checkbox."""
        if self.controller.callBackActive:
            self.controller.maskChange("highlights", self.checkboxHighlights.isChecked())

class HDRviewerView(QFrame):
    """A view for previewing HDR images."""

    def __init__(self, _controller=None, build=False):
        """Initialize the HDRviewerView.

        Args:
            _controller (optional): Reference to the controller. Defaults to None.
            build (bool, optional): Whether to build the view. Defaults to False.
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = _controller
        self.vbox = QVBoxLayout()
        self.hboxUp = QHBoxLayout()
        self.hboxDown = QHBoxLayout()

        self.label = QLabel("hdr preview")
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

    def reset(self):
        """Reset the HDR preview."""
        self.controller.displaySplash()

    def update(self):
        """Update the HDR preview."""
        self.controller.callBackUpdate()

    def compare(self):
        """Compare the HDR preview with the original image."""
        self.controller.callBackCompare()

    def auto(self):
        """Toggle the auto preview mode."""
        self.controller.callBackAuto(self.autoCheckBox.isChecked())

class LchColorSelectorView(QFrame):
    """A view for selecting and editing colors in LCH color space."""

    def __init__(self, _controller, defaultValues=None):
        """Initialize the LchColorSelectorView.

        Args:
            _controller: Reference to the controller
            defaultValues (optional): Default values for the color selectors. Defaults to None.
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = _controller
        self.vbox = QVBoxLayout()

        self.labelSelector = QLabel("Hue Chroma Lightness color selector")

        # Procedural image: Hue bar
        hueBarLch = hdrCore.image.Image.buildLchColorData((75, 75), (100, 100), (0, 360), (20, 720), width='h', height='c')
        hueBarRGB = hdrCore.processing.Lch_to_sRGB(hueBarLch, apply_cctf_encoding=True, clip=True)
        self.imageHueController = controller.ImageWidgetController()
        self.imageHueController.view.setMinimumSize(2, 72)
        self.imageHueController.setImage(hueBarRGB)
        hueBar2Lch = hdrCore.image.Image.buildLchColorData((75, 75), (100, 100), (0, 360), (20, 720), width='h', height='c')
        hueBar2RGB = hdrCore.processing.Lch_to_sRGB(hueBar2Lch, apply_cctf_encoding=True, clip=True)
        self.imageHueRangeController = controller.ImageWidgetController()
        self.imageHueRangeController.view.setMinimumSize(2, 72)
        self.imageHueRangeController.setImage(hueBarRGB)

        # Slider min for Hue
        self.sliderHueMin = QSlider(Qt.Horizontal)
        self.sliderHueMin.setRange(0, 360)
        self.sliderHueMin.setValue(0)
        self.sliderHueMin.setSingleStep(1)

        # Slider max for Hue
        self.sliderHueMax = QSlider(Qt.Horizontal)
        self.sliderHueMax.setRange(0, 360)
        self.sliderHueMax.setValue(360)
        self.sliderHueMax.setSingleStep(1)

        # Procedural image: Saturation bar
        saturationBarLch = hdrCore.image.Image.buildLchColorData((75, 75), (0, 100), (180, 180), (20, 720), width='c', height='L')
        saturationBarRGB = hdrCore.processing.Lch_to_sRGB(saturationBarLch, apply_cctf_encoding=True, clip=True)
        self.imageSaturationController = controller.ImageWidgetController()
        self.imageSaturationController.view.setMinimumSize(2, 72)
        self.imageSaturationController.setImage(saturationBarRGB)

        # Slider min for Chroma
        self.sliderChromaMin = QSlider(Qt.Horizontal)
        self.sliderChromaMin.setRange(0, 100)
        self.sliderChromaMin.setValue(0)
        self.sliderChromaMin.setSingleStep(1)

        # Slider max for Chroma
        self.sliderChromaMax = QSlider(Qt.Horizontal)
        self.sliderChromaMax.setRange(0, 100)
        self.sliderChromaMax.setValue(100)
        self.sliderChromaMax.setSingleStep(1)

        # Procedural image: Lightness bar
        lightnessBarLch = hdrCore.image.Image.buildLchColorData((0, 100), (0, 0), (180, 180), (20, 720), width='L', height='c')
        lightnessBarRGB = hdrCore.processing.Lch_to_sRGB(lightnessBarLch, apply_cctf_encoding=True, clip=True)
        self.imageLightnessController = controller.ImageWidgetController()
        self.imageLightnessController.view.setMinimumSize(2, 72)
        self.imageLightnessController.setImage(lightnessBarRGB)

        # Slider min for Lightness
        self.sliderLightMin = QSlider(Qt.Horizontal)
        self.sliderLightMin.setRange(0, 300)
        self.sliderLightMin.setValue(0)
        self.sliderLightMin.setSingleStep(1)

        # Slider max for Lightness
        self.sliderLightMax = QSlider(Qt.Horizontal)
        self.sliderLightMax.setRange(0, 300)
        self.sliderLightMax.setValue(300)
        self.sliderLightMax.setSingleStep(1)

        # Editor
        self.labelEditor = QLabel("color editor: hue shift, exposure, contrast, saturation")

        # Hue shift
        self.frameHueShift = QFrame()
        self.layoutHueShift = QHBoxLayout()
        self.frameHueShift.setLayout(self.layoutHueShift)
        self.sliderHueShift = QSlider(Qt.Horizontal)
        self.sliderHueShift.setRange(-180, +180)
        self.sliderHueShift.setValue(0)
        self.sliderHueShift.setSingleStep(1)
        self.valueHueShift = QLineEdit()
        self.valueHueShift.setText(str(0.0))
        self.layoutHueShift.addWidget(QLabel("hue shift"))
        self.layoutHueShift.addWidget(self.sliderHueShift)
        self.layoutHueShift.addWidget(self.valueHueShift)

        # Exposure
        self.frameExposure = QFrame()
        self.layoutExposure = QHBoxLayout()
        self.frameExposure.setLayout(self.layoutExposure)
        self.sliderExposure = QSlider(Qt.Horizontal)
        self.sliderExposure.setRange(-90, +90)
        self.sliderExposure.setValue(0)
        self.sliderExposure.setSingleStep(1)
        self.valueExposure = QLineEdit()
        self.valueExposure.setText(str(0.0))
        self.layoutExposure.addWidget(QLabel("exposure"))
        self.layoutExposure.addWidget(self.sliderExposure)
        self.layoutExposure.addWidget(self.valueExposure)

        # Contrast
        self.frameContrast = QFrame()
        self.layoutContrast = QHBoxLayout()
        self.frameContrast.setLayout(self.layoutContrast)
        self.sliderContrast = QSlider(Qt.Horizontal)
        self.sliderContrast.setRange(-100, +100)
        self.sliderContrast.setValue(0)
        self.sliderContrast.setSingleStep(1)
        self.valueContrast = QLineEdit()
        self.valueContrast.setText(str(0.0))
        self.layoutContrast.addWidget(QLabel("contrast"))
        self.layoutContrast.addWidget(self.sliderContrast)
        self.layoutContrast.addWidget(self.valueContrast)

        # Saturation
        self.frameSaturation = QFrame()
        self.layoutSaturation = QHBoxLayout()
        self.frameSaturation.setLayout(self.layoutSaturation)
        self.sliderSaturation = QSlider(Qt.Horizontal)
        self.sliderSaturation.setRange(-100, +100)
        self.sliderSaturation.setValue(0)
        self.sliderSaturation.setSingleStep(1)
        self.valueSaturation = QLineEdit()
        self.valueSaturation.setText(str(0.0))
        self.layoutSaturation.addWidget(QLabel("saturation"))
        self.layoutSaturation.addWidget(self.sliderSaturation)
        self.layoutSaturation.addWidget(self.valueSaturation)

        # Reset buttons
        self.resetSelection = QPushButton("reset selection")
        self.resetEdit = QPushButton("reset edit")

        # Mask checkbox
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
        self.vbox.addWidget(self.resetSelection)
        self.vbox.addWidget(self.labelEditor)
        self.vbox.addWidget(self.frameHueShift)
        self.vbox.addWidget(self.frameSaturation)
        self.vbox.addWidget(self.frameExposure)
        self.vbox.addWidget(self.frameContrast)
        self.vbox.addWidget(self.checkboxMask)
        self.vbox.addWidget(self.resetEdit)

        self.setLayout(self.vbox)

        # Callbacks
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
        self.resetSelection.clicked.connect(self.controller.resetSelection)
        self.resetEdit.clicked.connect(self.controller.resetEdit)

    def sliderHueChange(self):
        """Handle the change of the hue sliders."""
        hmin = self.sliderHueMin.value()
        hmax = self.sliderHueMax.value()

        # Redraw hue range and chroma bar
        hueRangeBarLch = hdrCore.image.Image.buildLchColorData((75, 75), (100, 100), (hmin, hmax), (20, 720), width='h', height='c')
        hueRangeBarRGB = hdrCore.processing.Lch_to_sRGB(hueRangeBarLch, apply_cctf_encoding=True, clip=True)
        self.imageHueRangeController.setImage(hueRangeBarRGB)
        saturationBarLch = hdrCore.image.Image.buildLchColorData((75, 75), (0, 100), (hmin, hmax), (20, 720), width='c', height='L')
        saturationBarRGB = hdrCore.processing.Lch_to_sRGB(saturationBarLch, apply_cctf_encoding=True, clip=True)
        self.imageSaturationController.setImage(saturationBarRGB)

        # Call controller
        self.controller.sliderHueChange(hmin, hmax)

    def sliderChromaChange(self):
        """Handle the change of the chroma sliders."""
        vmin = self.sliderChromaMin.value()
        vmax = self.sliderChromaMax.value()
        # Call controller
        self.controller.sliderChromaChange(vmin, vmax)

    def sliderLightnessChange(self):
        """Handle the change of the lightness sliders."""
        vmin = self.sliderLightMin.value() / 3.0
        vmax = self.sliderLightMax.value() / 3.0
        # Call controller
        self.controller.sliderLightnessChange(vmin, vmax)

    def sliderExposureChange(self):
        """Handle the change of the exposure slider."""
        ev = round(self.sliderExposure.value() / 30, 1)
        self.valueExposure.setText(str(ev))
        self.controller.sliderExposureChange(ev)

    def sliderSaturationChange(self):
        """Handle the change of the saturation slider."""
        ev = self.sliderSaturation.value()
        self.valueSaturation.setText(str(ev))
        self.controller.sliderSaturationChange(ev)

    def sliderContrastChange(self):
        """Handle the change of the contrast slider."""
        ev = self.sliderContrast.value()
        self.valueContrast.setText(str(ev))
        self.controller.sliderContrastChange(ev)

    def sliderHueShiftChange(self):
        """Handle the change of the hue shift slider."""
        hs = self.sliderHueShift.value()
        self.valueHueShift.setText(str(hs))
        self.controller.sliderHueShiftChange(hs)

    def checkboxMaskChange(self):
        """Handle the change of the mask checkbox."""
        self.controller.checkboxMaskChange(self.checkboxMask.isChecked())

class GeometryView(QFrame):
    """A view for adjusting the geometry of an image."""

    def __init__(self, _controller):
        """Initialize the GeometryView.

        Args:
            _controller: Reference to the controller
        """
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.controller = _controller
        self.vbox = QVBoxLayout()

        # Cropping adjustment
        self.frameCroppingVerticalAdjustement = QFrame()
        self.layoutCroppingVerticalAdjustement = QHBoxLayout()
        self.frameCroppingVerticalAdjustement.setLayout(self.layoutCroppingVerticalAdjustement)
        self.sliderCroppingVerticalAdjustement = QSlider(Qt.Horizontal)
        self.sliderCroppingVerticalAdjustement.setRange(-100, +100)
        self.sliderCroppingVerticalAdjustement.setValue(0)
        self.sliderCroppingVerticalAdjustement.setSingleStep(1)
        self.valueCroppingVerticalAdjustement = QLineEdit()
        self.valueCroppingVerticalAdjustement.setText(str(0.0))
        self.layoutCroppingVerticalAdjustement.addWidget(QLabel("cropping adj."))
        self.layoutCroppingVerticalAdjustement.addWidget(self.sliderCroppingVerticalAdjustement)
        self.layoutCroppingVerticalAdjustement.addWidget(self.valueCroppingVerticalAdjustement)

        # Rotation
        self.frameRotation = QFrame()
        self.layoutRotation = QHBoxLayout()
        self.frameRotation.setLayout(self.layoutRotation)
        self.sliderRotation = QSlider(Qt.Horizontal)
        self.sliderRotation.setRange(-60, +60)
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

    def sliderCroppingVerticalAdjustementChange(self):
        """Handle the change of the cropping vertical adjustment slider."""
        v = self.sliderCroppingVerticalAdjustement.value()
        self.valueCroppingVerticalAdjustement.setText(str(v))
        self.controller.sliderCroppingVerticalAdjustementChange(v)

    def sliderRotationChange(self):
        """Handle the change of the rotation slider."""
        v = self.sliderRotation.value() / 6
        self.valueRotation.setText(str(v))
        self.controller.sliderRotationChange(v)

class ImageAestheticsView(QSplitter):
    """A view for displaying and adjusting the aesthetics of an image."""

    def __init__(self, _controller, build=False):
        """Initialize the ImageAestheticsView.

        Args:
            _controller: Reference to the controller
            build (bool, optional): Whether to build the view. Defaults to False.
        """
        if pref.verbose:
            print(" [VIEW] >> AestheticsImageView.__init__()")
        super().__init__(Qt.Vertical)
        self.controller = _controller
        self.imageWidgetController = controller.ImageWidgetController()
        self.layout = QVBoxLayout()

        # Color palette: node selector(node name), color number, palette image.
        self.labelColorPalette = QLabel("color palette")
        self.labelNodeSelector = QLabel("process output:")
        self.nodeSelector = QComboBox(self)

        # Recover process nodes names from buildProcessPipe
        processNodeNameList = []
        emptyProcessPipe = model.EditImageModel.buildProcessPipe()
        for node in emptyProcessPipe.processNodes:
            processNodeNameList.append(node.name)

        # Add 'output' at the end to help user
        processNodeNameList.append('output')
        self.nodeSelector.addItems(processNodeNameList)
        self.nodeSelector.setCurrentIndex(len(processNodeNameList) - 1)

        # QSpinBox
        self.labelColorsNumber = QLabel("number of colors:")
        self.nbColors = QSpinBox(self)
        self.nbColors.setRange(2, 8)
        self.nbColors.setValue(5)

        self.paletteImageWidgetController = controller.ImageWidgetController()
        imgPalette = hdrCore.aesthetics.Palette('defaultLab5', np.linspace([0, 0, 0], [100, 0, 0], 5), hdrCore.image.ColorSpace.build('Lab'), hdrCore.image.imageType.SDR).createImageOfPalette()
        self.paletteImageWidgetController.setImage(imgPalette)
        self.paletteImageWidgetController.view.setMinimumSize(40, 10)

        # Add widgets to layout
        self.layout.addWidget(self.labelColorPalette)
        self.layout.addWidget(self.labelNodeSelector)
        self.layout.addWidget(self.nodeSelector)
        self.layout.addWidget(self.labelColorsNumber)
        self.layout.addWidget(self.nbColors)
        self.layout.addWidget(self.paletteImageWidgetController.view)

        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        # Scroll and etc.
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QLabel()
        self.container.setLayout(self.layout)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        # Add widget to QSplitter
        self.addWidget(self.imageWidgetController.view)
        self.addWidget(self.scroll)
        self.setSizes([60, 40])

    def setProcessPipe(self, processPipe, paletteImg):
        """Set the process pipe and update the view.

        Args:
            processPipe: The process pipe to set
            paletteImg: The palette image to display
        """
        self.imageWidgetController.setImage(processPipe.getImage())
        self.paletteImageWidgetController.setImage(paletteImg)

class ColorEditorsAutoView(QPushButton):
    """A button view for automatic color editors."""

    def __init__(self, controller):
        """Initialize the ColorEditorsAutoView.

        Args:
            controller: Reference to the controller
        """
        super().__init__("auto color selection [! reset edit]")
        self.controller = controller
        self.clicked.connect(self.controller.auto)

# ------------------------------------------------------------------------------------------
