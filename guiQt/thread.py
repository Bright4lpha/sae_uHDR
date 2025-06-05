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
import copy, time, random
import hdrCore
from . import model
from PyQt5.QtCore import QRunnable, Qt, QThreadPool
from timeit import default_timer as timer
import preferences.preferences as pref

# -----------------------------------------------------------------------------
# --- Class RequestCompute ----------------------------------------------------
# -----------------------------------------------------------------------------
class RequestCompute(object):
    """
    Manage parallel (multithreading) computation of processpipe.compute() when editing image (not used for display HDR or export HDR):
    - Uses a single/specific thread to compute process-pipe.
    - Stores compute request when user changes editing values, restarts process-pipe computing when the previous one is finished.

    Attributes:
        parent (guiQt.model.EditImageModel): Reference to parent, used to callback parent when processing is over.
        requestDict (dict): Dictionary that stores editing values.
        pool (QThreadPool): Qt thread pool.
        processpipe (hdrCore.processing.ProcessPipe): Active process pipe.
        readyToRun (bool): True when no processing is ongoing, else False.
        waitingUpdate (bool): True if requestCompute has been called during a processing.

    Methods:
        setProcessPipe: Set the current active process pipe.
        requestCompute: Send new parameters for a process-node and request a new process pipe computation.
        endCompute: Called when process-node computation is finished.
    """

    def __init__(self, parent):
        """
        Initialize the RequestCompute.

        Args:
            parent: Reference to the parent model.
        """
        self.parent = parent
        self.requestDict = {}  # Store requestCompute key: processNodeId, value: processNode params
        self.pool = QThreadPool.globalInstance()  # Get global pool
        self.processpipe = None  # Process pipe reference
        self.readyToRun = True
        self.waitingUpdate = False

    def setProcessPipe(self, pp):
        """
        Set the current active process pipe.

        Args:
            pp (hdrCore.processing.ProcessPipe): The process pipe to set.
        """
        self.processpipe = pp

    def requestCompute(self, id, params):
        """
        Send new parameters for a process-node and request a new process pipe computation.

        Args:
            id (int): Index of process-node in process pipe.
            params (dict): Parameters of process-node.
        """
        self.requestDict[id] = copy.deepcopy(params)

        if self.readyToRun:
            # Start processing process pipe
            self.pool.start(RunCompute(self))
        else:
            # If a computation is already running
            self.waitingUpdate = True

    def endCompute(self):
        """
        Called when process-node computation is finished.
        Get processed image and send it to parent (guiQt.model.EditImageModel).
        If there are new requestCompute, restart computation of process pipe.
        """
        imgTM = self.processpipe.getImage(toneMap=True)
        self.parent.updateImage(imgTM)
        if self.waitingUpdate:
            self.pool.start(RunCompute(self))
            self.waitingUpdate = False

class RunCompute(QRunnable):
    """
    Defines the run method that executes on a dedicated thread: process pipe computation.

    Attributes:
        parent (guiQt.thread.RequestCompute): Parent called endCompute() when processing is over.

    Methods:
        run: Method called by the Qt Thread pool.
    """

    def __init__(self, parent):
        """
        Initialize the RunCompute.

        Args:
            parent: Reference to the parent RequestCompute.
        """
        super().__init__()
        self.parent = parent

    def run(self):
        """
        Method called by the Qt Thread pool.
        Calls parent.endCompute() when the process is over.
        """
        self.parent.readyToRun = False
        for k in self.parent.requestDict.keys():
            self.parent.processpipe.setParameters(k, self.parent.requestDict[k])
        cpp = True
        if cpp:
            img = copy.deepcopy(self.parent.processpipe.getInputImage())
            imgRes = hdrCore.coreC.coreCcompute(img, self.parent.processpipe)
            self.parent.processpipe.setOutput(imgRes)
            self.parent.readyToRun = True
            self.parent.endCompute()
        else:
            start = timer()
            self.parent.processpipe.compute()
            dt = timer() - start
            self.parent.readyToRun = True
            self.parent.endCompute()

class RequestLoadImage(object):
    """
    Manage parallel (multithreading) computation of loading images:
    - Uses a new thread to load each image.
    - Calls parent with process-pipe associated to loaded image.

    Attributes:
        parent (guiQt.model.ImageGalleryModel): Reference to parent, used to callback parent when processing is over.
        pool (QThreadPool): Qt thread pool.
        requestsDone (dict): Key is index of image in page. requestsDone[requestsDone] = True when image is loaded.

    Methods:
        requestLoad: Request to load an image.
        endLoadImage: Called when loading is over or failed (IOError, ValueError).
    """

    def __init__(self, parent):
        """
        Initialize the RequestLoadImage.

        Args:
            parent: Reference to the parent model.
        """
        self.parent = parent
        self.pool = QThreadPool.globalInstance()  # Get a global pool
        self.requestsDone = {}

    def requestLoad(self, minIdxInPage, imgIdxInPage, filename):
        """
        Request to load an image.

        Args:
            minIdxInPage (int): Image/process pipe index of first image in page.
            imgIdxInPage (int): Index of image/process pipe in the current page.
            filename (str): Image filename.
        """
        self.requestsDone[minIdxInPage + imgIdxInPage] = False
        self.pool.start(RunLoadImage(self, minIdxInPage, imgIdxInPage, filename))

    def endLoadImage(self, error, idx0, idx, processPipe, filename):
        """
        Called when loading is over or failed (IOError, ValueError).
        Set process-pipe into parent (guiQt.model.ImageGalleryModel) then update view.
        If loading failed (IOError, ValueError) recall self.requestLoad().

        Args:
            error (bool): True if loading failed (take into account ValueError).
            idx0 (int): Image/process pipe index of first image in page.
            idx (int): Index of image/process pipe in the current page.
            processPipe (hdrCore.processing.ProcessPipe): Process-pipe associated to loaded image.
            filename (str): Filename of image.
        """
        if not error:
            self.requestsDone[idx0 + idx] = True
            self.parent.processPipes[idx0 + idx] = processPipe
            self.parent.controller.view.updateImage(idx, processPipe, filename)
        else:
            self.requestLoad(idx0, idx, filename)

class RunLoadImage(QRunnable):
    """
    Defines the run method that executes on a dedicated thread: image loading.

    Attributes:
        parent (guiQt.thread.RequestLoadImage): Parent called endLoadImage() when processing is over.
        minIdxInPage (int): Image/process pipe index of first image in page.
        imgIdxInPage (int): Index of image/process pipe in the current page.
        filename (str): Filename of the image to load.
    """

    def __init__(self, parent, minIdxInPage, imgIdxInPage, filename):
        """
        Initialize the RunLoadImage.

        Args:
            parent: Reference to the parent RequestLoadImage.
            minIdxInPage (int): Image/process pipe index of first image in page.
            imgIdxInPage (int): Index of image/process pipe in the current page.
            filename (str): Filename of the image to load.
        """
        super().__init__()
        self.parent = parent
        self.minIdxInPage = minIdxInPage
        self.imgIdxInPage = imgIdxInPage
        self.filename = filename

    def run(self):
        """
        Method called by the Qt Thread pool.
        Calls parent.endLoadImage() when the process is over.
        """
        try:
            image_ = hdrCore.image.Image.read(self.filename, thumb=True)
            processPipe = model.EditImageModel.buildProcessPipe()
            processPipe.setImage(image_)
            processPipe.compute()
            self.parent.endLoadImage(False, self.minIdxInPage, self.imgIdxInPage, processPipe, self.filename)
        except (IOError, ValueError) as e:
            self.parent.endLoadImage(True, self.minIdxInPage, self.imgIdxInPage, None, self.filename)

class pCompute(object):
    """
    Manage parallel (multithreading) computation of processpipe.compute when displaying HDR image or exporting HDR image:
    - The image is split into multiple parts (called splits), then multithreading processing is started for each split.
    - When all computations of the splits are over, the processed splits are merged (note that geometry processing computation is processed after merging).
    - The parent callback function is called with the processed image (tone-mapped or not according to constructor parameters).

    Attributes:
        callBack (function): Function called when processing is over.
        progress (function): Function called to display processing progress.
        nbSplits (int): Number of image splits.
        nbDone (int): Number of splits for which the computation is over.
        geometryNode (hdrCore.process.ProcessNode): Geometry process node which computation is done at the end.
        meta (hdrCore.metadata.metadata): Metadata of process pipe input image.

    Methods:
        endCompute: Called when a split computation is finished.
    """

    def __init__(self, callBack, processpipe, nbWidth, nbHeight, toneMap=True, progress=None, meta=None):
        """
        Initialize the pCompute.

        Args:
            callBack (function): Function called when processing is over.
            processpipe: The process pipe to compute.
            nbWidth (int): Number of horizontal splits.
            nbHeight (int): Number of vertical splits.
            toneMap (bool, optional): Whether to tone map the image. Defaults to True.
            progress (function, optional): Function called to display processing progress. Defaults to None.
            meta (hdrCore.metadata.metadata, optional): Metadata of process pipe input image. Defaults to None.
        """
        self.callBack = callBack
        self.progress = progress
        self.nbSplits = nbWidth * nbHeight
        self.nbDone = 0
        self.geometryNode = None
        self.meta = meta
        # Recover and split image
        input = processpipe.getInputImage()

        # Store last process node (geometry) and remove it from process pipe
        if isinstance(processpipe.processNodes[-1].process, hdrCore.processing.geometry):
            self.geometryNode = copy.deepcopy(processpipe.processNodes[-1])

            # Remove geometry node (the last one)
            processpipe.processNodes = processpipe.processNodes[:-1]

        # Split image and store split images
        self.splits = input.split(nbWidth, nbHeight)

        self.pool = QThreadPool.globalInstance()

        # Duplicate process pipe, set image split and start
        for idxY, line in enumerate(self.splits):
            for idxX, split in enumerate(line):
                pp = copy.deepcopy(processpipe)
                pp.setImage(split)
                # Start compute
                self.pool.start(pRun(self, pp, toneMap, idxX, idxY))

    def endCompute(self, idx, idy, split):
        """
        Called when a split computation is finished.

        Args:
            idx (int): X index of the split.
            idy (int): Y index of the split.
            split: The processed split.
        """
        self.splits[idy][idx] = copy.deepcopy(split)
        self.nbDone += 1
        if self.progress:
            percent = str(int(self.nbDone * 100 / self.nbSplits)) + '%'
            self.progress('HDR image process-pipe computation:' + percent)
        if self.nbDone == self.nbSplits:
            res = hdrCore.image.Image.merge(self.splits)
            # Process geometry
            if self.geometryNode:
                res = self.geometryNode.process.compute(res, **self.geometryNode.params)
            # Call back caller
            self.callBack(res, self.meta)

class pRun(QRunnable):
    """
    Defines the run method that executes on a dedicated thread: split computation.

    Attributes:
        parent (guiQt.thread.pCompute): Parent called endCompute() when processing is over.
        processpipe: The process pipe to compute.
        idxX (int): X index of the split.
        idxY (int): Y index of the split.
        toneMap (bool): Whether to tone map the image.
    """

    def __init__(self, parent, processpipe, toneMap, idxX, idxY):
        """
        Initialize the pRun.

        Args:
            parent: Reference to the parent pCompute.
            processpipe: The process pipe to compute.
            toneMap (bool): Whether to tone map the image.
            idxX (int): X index of the split.
            idxY (int): Y index of the split.
        """
        super().__init__()
        self.parent = parent
        self.processpipe = processpipe
        self.idx = (idxX, idxY)
        self.toneMap = toneMap

    def run(self):
        """
        Method called by the Qt Thread pool.
        Calls parent.endCompute() when the process is over.
        """
        self.processpipe.compute()
        pRes = self.processpipe.getImage(toneMap=self.toneMap)
        self.parent.endCompute(self.idx[0], self.idx[1], pRes)

class cCompute(object):
    """
    Manage parallel (multithreading) computation of processpipe.compute.
    """

    def __init__(self, callBack, processpipe, toneMap=True, progress=None):
        """
        Initialize the cCompute.

        Args:
            callBack (function): Function called when processing is over.
            processpipe: The process pipe to compute.
            toneMap (bool, optional): Whether to tone map the image. Defaults to True.
            progress (function, optional): Function called to display processing progress. Defaults to None.
        """
        self.callBack = callBack
        self.progress = progress

        # Recover image
        input = processpipe.getInputImage()

        self.pool = QThreadPool.globalInstance()
        self.pool.start(cRun(self, processpipe, toneMap))

    def endCompute(self, img):
        """
        Called when the computation is finished.

        Args:
            img: The processed image.
        """
        self.callBack(img)

class cRun(QRunnable):
    """
    Defines the run method that executes on a dedicated thread: process pipe computation.

    Attributes:
        parent (guiQt.thread.cCompute): Parent called endCompute() when processing is over.
        processpipe: The process pipe to compute.
        toneMap (bool): Whether to tone map the image.
    """

    def __init__(self, parent, processpipe, toneMap):
        """
        Initialize the cRun.

        Args:
            parent: Reference to the parent cCompute.
            processpipe: The process pipe to compute.
            toneMap (bool): Whether to tone map the image.
        """
        super().__init__()
        self.parent = parent
        self.processpipe = processpipe
        self.toneMap = toneMap

    def run(self):
        """
        Method called by the Qt Thread pool.
        Calls parent.endCompute() when the process is over.
        """
        img = copy.deepcopy(self.processpipe.getInputImage())
        imgRes = hdrCore.coreC.coreCcompute(img, self.processpipe)
        self.processpipe.setOutput(imgRes)

        pRes = self.processpipe.getImage(toneMap=self.toneMap)
        self.parent.endCompute(pRes)

class RequestAestheticsCompute(object):
    """
    Manage parallel (multithreading) computation of processpipe.compute() when editing image (not used for display HDR or export HDR):
    - Uses a single/specific thread to compute process-pipe.
    - Stores compute request when user changes editing values, restarts process-pipe computing when the previous one is finished.

    Attributes:
        parent (guiQt.model.EditImageModel): Reference to parent, used to callback parent when processing is over.
        requestDict (dict): Dictionary that stores editing values.
        pool (QThreadPool): Qt thread pool.
        processpipe (hdrCore.processing.ProcessPipe): Active process pipe.
        readyToRun (bool): True when no processing is ongoing, else False.
        waitingUpdate (bool): True if requestCompute has been called during a processing.

    Methods:
        setProcessPipe: Set the current active process pipe.
        requestCompute: Send new parameters for a process-node and request a new process pipe computation.
        endCompute: Called when process-node computation is finished.
    """

    def __init__(self, parent):
        """
        Initialize the RequestAestheticsCompute.

        Args:
            parent: Reference to the parent model.
        """
        self.parent = parent
        self.requestDict = {}  # Store requestCompute key: processNodeId, value: processNode params
        self.pool = QThreadPool.globalInstance()  # Get global pool
        self.processpipe = None  # Process pipe reference
        self.readyToRun = True
        self.waitingUpdate = False

    def setProcessPipe(self, pp):
        """
        Set the current active process pipe.

        Args:
            pp (hdrCore.processing.ProcessPipe): The process pipe to set.
        """
        self.processpipe = pp

    def requestCompute(self, id, params):
        """
        Send new parameters for a process-node and request a new process pipe computation.

        Args:
            id (int): Index of process-node in process pipe.
            params (dict): Parameters of process-node.
        """
        self.requestDict[id] = copy.deepcopy(params)

        if self.readyToRun:
            # Start processing process pipe
            self.pool.start(RunAestheticsCompute(self))
        else:
            # If a computation is already running
            self.waitingUpdate = True

    def endCompute(self):
        """
        Called when process-node computation is finished.
        Get processed image and send it to parent (guiQt.model.EditImageModel).
        If there are new requestCompute, restart computation of process pipe.
        """
        imgTM = self.processpipe.getImage(toneMap=True)
        self.parent.updateImage(imgTM)
        if self.waitingUpdate:
            self.pool.start(RunAestheticsCompute(self))
            self.waitingUpdate = False

class RunAestheticsCompute(QRunnable):
    """
    Defines the run method that executes on a dedicated thread: process pipe computation.

    Attributes:
        parent (guiQt.thread.RequestCompute): Parent called endCompute() when processing is over.
    """

    def __init__(self, parent):
        """
        Initialize the RunAestheticsCompute.

        Args:
            parent: Reference to the parent RequestAestheticsCompute.
        """
        super().__init__()
        self.parent = parent

    def run(self):
        """
        Method called by the Qt Thread pool.
        Calls parent.endCompute() when the process is over.
        """
        self.parent.readyToRun = False
        for k in self.parent.requestDict.keys():
            self.parent.processpipe.setParameters(k, self.parent.requestDict[k])
        start = timer()
        self.parent.processpipe.compute()
        dt = timer() - start
        self.parent.readyToRun = True
        self.parent.endCompute()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
