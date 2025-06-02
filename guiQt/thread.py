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
    manage parallel (multithreading) computation of processpipe.compute() when editing image (not used for display HDR or export HDR):
        - uses a single/specific thread to compute process-pipe,
        - store compute request when user changes editing values, restart process-pipe computing when the previous one is finished.

    Attributes:
        parent (guiQt.model.EditImageModel): reference to parent, used to callback parent when processing is over.
        requestDict (dict): dict that stores editing values.
        pool (QThreadPool): Qt thread pool.
        processpipe (hdrCore.processing.Processpipe): active processpipe.
        readyToRun (bool): True when no processing is ongoing, else False.
        waitingUpdate (bool): True if requestCompute has been called during a processing.

    Methods:
        setProcessPipe
        requestCompute
        endCompute
    """

    def __init__(self, parent):

        self.parent = parent

        self.requestDict= {} # store resqustCompute key:processNodeId, value: processNode params

        self.pool = QThreadPool.globalInstance()        # get global pool
        self.processpipe = None                         # processpipe ref

        self.readyToRun = True
        self.waitingUpdate = False

    def setProcessPipe(self,pp):
        """set the current active processpipe.

            Args:
                pp (hrdCore.processing.ProcessPipe, Required)

            Returns:
                
        """
        self.processpipe = pp
    
    def requestCompute(self, id, params):
        """send new parameters for a process-node and request a new processpipe computation.

            Args:
                id (int, Required): index of process-node in processpipe
                params (dict, Required): parameters of process-node 

            Returns:
                
        """
        self.requestDict[id] = copy.deepcopy(params)


        if self.readyToRun:
            # start processing processpipe
            self.pool.start(RunCompute(self))
        else:
            # if a computation is already running
            self.waitingUpdate = True

    def endCompute(self):
        """called when process-node computation is finished.
            Get processed image and send it to parent (guiQt.model.EditImageModel).
            If there are new requestCompute, restart computation of processpipe

        Args:

        Retruns:

        """
        imgTM = self.processpipe.getImage(toneMap=True)
        self.parent.updateImage(imgTM)
        if self.waitingUpdate:
            self.pool.start(RunCompute(self))
            self.waitingUpdate = False
# -----------------------------------------------------------------------------
# --- Class RunCompute --------------------------------------------------------
# -----------------------------------------------------------------------------
class RunCompute(QRunnable):
    """defines the run method that executes on a dedicated thread: processpipe computation.
    
        Attributes:
            parent (guiQt.thread.RequestCompute): parent called.endCompute() when processing is over.

        Methods:
            run
    """
    def __init__(self,parent):
        super().__init__()
        self.parent = parent

    def run(self):
        """method called by the Qt Thread pool.
            Calls parent.endCompute() when process is over.

            Args:

            Returns:

        """
        self.parent.readyToRun = False
        for k in self.parent.requestDict.keys(): self.parent.processpipe.setParameters(k,self.parent.requestDict[k])
        cpp = True
        if cpp:
            img  = copy.deepcopy(self.parent.processpipe.getInputImage())
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
# -----------------------------------------------------------------------------
# --- Class RequestLoadImage --------------------------------------------------
# -----------------------------------------------------------------------------
class RequestLoadImage(object):
    """
    manage parallel (multithreading) computation of loading images:
        - uses a new thread to load each image.
        - calls parent with process-pipe associated to loaded image
    Attributes:
        parent (guiQt.model.ImageGalleryModel): reference to parent, used to callback parent when processing is over.
        pool (QThreadPool): Qt thread pool.
        requestsDone (Dict): key is index of image in page 
            requestsDone[requestsDone]= True when image is loaded

    Methods:
        requestLoad
        endLoadImage

    """

    def __init__(self, parent):

        self.parent = parent
        self.pool = QThreadPool.globalInstance()        # get a global pool
        self.requestsDone = {}

    def requestLoad(self, minIdxInPage, imgIdxInPage, filename):
        """
        Args:
            minIdxInPage (int, Required): image/processpipe index of first image in page.
            imgIdxInPage (int, Required): index of image/processpipe in the current page. 
            filename     (str, Required): image filename.

        Returns:
            
        """
        self.requestsDone[minIdxInPage+ imgIdxInPage] = False
        self.pool.start(RunLoadImage(self,minIdxInPage, imgIdxInPage,filename))

    def endLoadImage(self,error,idx0, idx,processPipe, filename):
        """called when loading is over or failed (IOError, ValueError).
            Set process-pipe into parent (guiQt.model.ImageGalleryModel) then update view.
            If loading failed (IOError, ValueError) recall self.requestLoad()

        Args:
            error           (bool, Required): True if loading failed (take into account ValueError).
            idx0            (int, Required): image/processpipe index of first image in page.
            idx             (int, Required): index of image/processpipe in the current page.
            processPipe     (hrdCore.processing.ProcessPipe, Required):  process-pipe associated to loaded image.
            filename        (str, Required): filename of image

        Returns:
        """
        if not error:
            self.requestsDone[idx0 + idx] = True
            self.parent.processPipes[idx0 + idx]= processPipe
            self.parent.controller.view.updateImage(idx,processPipe, filename)
        else:
            self.requestLoad(idx0, idx, filename)
# -----------------------------------------------------------------------------
# --- Class RunLoadImage ------------------------------------------------------
# -----------------------------------------------------------------------------
class RunLoadImage(QRunnable):
    def __init__(self,parent, minIdxInPage, imgIdxInPage, filename):
        """
        Args:

        Returns:
        """
        super().__init__()
        self.parent = parent
        self.minIdxInPage = minIdxInPage
        self.imgIdxInPage = imgIdxInPage
        self.filename = filename

    def run(self):
        """
        Args:

        Returns:
        """
        try:
            image_ = hdrCore.image.Image.read(self.filename, thumb=True)
            processPipe = model.EditImageModel.buildProcessPipe()
            processPipe.setImage(image_)                      
            processPipe.compute()
            self.parent.endLoadImage(False, self.minIdxInPage, self.imgIdxInPage, processPipe, self.filename)
        except(IOError, ValueError) as e:
            self.parent.endLoadImage(True, self.minIdxInPage, self.imgIdxInPage, None, self.filename)
# -----------------------------------------------------------------------------
# --- Class pCompute ----------------------------------------------------------
# -----------------------------------------------------------------------------
class pCompute(object):
    """
    manage parallel (multithreading) computation of processpipe.compute when display HDR image or export HDR image:
        - the image is split into multiple parts (called splits), then multithreading processing is started for each split,
        - when all  computations of the splits are over, the processed splits are merge (note that geometry processing computation is processed after merging)
        - the parent callback function is called with the processed image (tone-mapped or not according to constructor parameters)

    Attributes:
        callBack (function): function called when processing is over.
        progress (function): function called to display processing progress.
        nbSplits (int): number of image splits
        nbDone (int): for which the computation is over. 
        geometryNode (hdrCore.process.ProcessNode): geometry process node which compuation is done at the end.
        meta (hdrCore.metadata.metadata): metadata of processpipe input image.

    Methods:
        endCompute
    """

    def __init__(self, callBack, processpipe,nbWidth,nbHeight, toneMap=True, progress=None, meta=None):
        self.callBack = callBack
        self.progress =progress
        self.nbSplits = nbWidth*nbHeight
        self.nbDone = 0
        self.geometryNode = None
        self.meta = meta
        # recover and split image
        input =  processpipe.getInputImage()

        # store last processNode (geometry) and remove it from processpipe
        if isinstance(processpipe.processNodes[-1].process,hdrCore.processing.geometry):
            self.geometryNode = copy.deepcopy(processpipe.processNodes[-1]) 

            # remove geometry node (the last one) 
            processpipe.processNodes = processpipe.processNodes[:-1]
       
        # split image and store splited images
        self.splits = input.split(nbWidth,nbHeight)

        self.pool = QThreadPool.globalInstance() 

        # duplicate processpipe, set image split and start
        for idxY,line in enumerate(self.splits):
            for idxX,split in enumerate(line):
                pp = copy.deepcopy(processpipe)
                pp.setImage(split)
                # start compute
                self.pool.start(pRun(self,pp,toneMap,idxX,idxY))

    def endCompute(self,idx,idy, split):
        """
        Args:

        Returns:
        
        """
        self.splits[idy][idx]= copy.deepcopy(split)
        self.nbDone += 1
        if self.progress:
            percent = str(int(self.nbDone*100/self.nbSplits))+'%'
            self.progress('HDR image process-pipe computation:'+percent)
        if self.nbDone == self.nbSplits:
            res = hdrCore.image.Image.merge(self.splits)
            # process geometry
            if self.geometryNode:
                res = self.geometryNode.process.compute(res,**self.geometryNode.params)
            # callBack caller
            self.callBack(res, self.meta)
# -----------------------------------------------------------------------------
# --- Class pRun --------------------------------------------------------------
# -----------------------------------------------------------------------------
class pRun(QRunnable):
    """
        Args:

        Returns:
        
    """
    def __init__(self,parent,processpipe,toneMap, idxX,idxY):
        """
        """
        super().__init__()
        self.parent = parent
        self.processpipe = processpipe
        self.idx = (idxX,idxY)
        self.toneMap = toneMap

    def run(self):
        """
        Args:

        Returns:
        
        """
        self.processpipe.compute()
        pRes = self.processpipe.getImage(toneMap=self.toneMap)
        self.parent.endCompute(self.idx[0],self.idx[1], pRes)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Class cCompute ----------------------------------------------------------
# -----------------------------------------------------------------------------
class cCompute(object):
    """xxx
    """

    def __init__(self, callBack, processpipe, toneMap=True, progress=None):
        self.callBack = callBack
        self.progress =progress

        # recover image
        input =  processpipe.getInputImage()

        self.pool = QThreadPool.globalInstance() 
        self.pool.start(cRun(self,processpipe,toneMap))

    def endCompute(self, img):
        """
        Args:

        Returns:
        
        """
        self.callBack(img)
# -----------------------------------------------------------------------------
# --- Class cRun --------------------------------------------------------------
# -----------------------------------------------------------------------------
class cRun(QRunnable):
    """
        Args:

        Returns:
        
    """
    def __init__(self,parent,processpipe,toneMap):
        """
        """
        super().__init__()
        self.parent = parent
        self.processpipe = processpipe
        self.toneMap = toneMap

    def run(self):
        """
        Args:

        Returns:
        
        """
        img  = copy.deepcopy(self.processpipe.getInputImage())
        imgRes = hdrCore.coreC.coreCcompute(img, self.processpipe)
        self.processpipe.setOutput(imgRes)

        pRes = self.processpipe.getImage(toneMap=self.toneMap)
        self.parent.endCompute(pRes)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Class RequestAestheticsCompute ----------------------------------------------------
# -----------------------------------------------------------------------------
class RequestAestheticsCompute(object):
    """
    manage parallel (multithreading) computation of processpipe.compute() when editing image (not used for display HDR or export HDR):
        - uses a single/specific thread to compute process-pipe,
        - store compute request when user changes editing values, restart process-pipe computing when the previous one is finished.

    Attributes:
        parent (guiQt.model.EditImageModel): reference to parent, used to callback parent when processing is over.
        requestDict (dict): dict that stores editing values.
        pool (QThreadPool): Qt thread pool.
        processpipe (hdrCore.processing.Processpipe): active processpipe.
        readyToRun (bool): True when no processing is ongoing, else False.
        waitingUpdate (bool): True if requestCompute has been called during a processing.

    Methods:
        setProcessPipe
        requestCompute
        endCompute
    """

    def __init__(self, parent):

        self.parent = parent

        self.requestDict= {} # store resqustCompute key:processNodeId, value: processNode params

        self.pool = QThreadPool.globalInstance()        # get global pool
        self.processpipe = None                         # processpipe ref

        self.readyToRun = True
        self.waitingUpdate = False

    def setProcessPipe(self,pp):
        """set the current active processpipe.

            Args:
                pp (hrdCore.processing.ProcessPipe, Required)

            Returns:
                
        """
        self.processpipe = pp
    
    def requestCompute(self, id, params):
        """send new parameters for a process-node and request a new processpipe computation.

            Args:
                id (int, Required): index of process-node in processpipe
                params (dict, Required): parameters of process-node 

            Returns:
                
        """
        self.requestDict[id] = copy.deepcopy(params)

        if self.readyToRun:
            # start processing processpipe
            self.pool.start(RunAestheticsCompute(self))
        else:
            # if a computation is already running
            self.waitingUpdate = True

    def endCompute(self):
        """called when process-node computation is finished.
            Get processed image and send it to parent (guiQt.model.EditImageModel).
            If there are new requestCompute, restart computation of processpipe

        Args:

        Retruns:

        """
        imgTM = self.processpipe.getImage(toneMap=True)
        self.parent.updateImage(imgTM)
        if self.waitingUpdate:
            self.pool.start(RunAestheticsCompute(self))
            self.waitingUpdate = False
# -----------------------------------------------------------------------------
# --- Class RunCompute --------------------------------------------------------
# -----------------------------------------------------------------------------
class RunAestheticsCompute(QRunnable):
    """defines the run method that executes on a dedicated thread: processpipe computation.
    
        Attributes:
            parent (guiQt.thread.RequestCompute): parent called.endCompute() when processing is over.

        Methods:
            run
    """
    def __init__(self,parent):
        super().__init__()
        self.parent = parent

    def run(self):
        """method called by the Qt Thread pool.
            Calls parent.endCompute() when process is over.

            Args:

            Returns:

        """
        self.parent.readyToRun = False
        for k in self.parent.requestDict.keys(): self.parent.processpipe.setParameters(k,self.parent.requestDict[k])
        start = timer()
        self.parent.processpipe.compute()
        dt = timer() - start
        self.parent.readyToRun = True
        self.parent.endCompute()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
