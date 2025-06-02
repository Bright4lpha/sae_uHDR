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
package hdrCoreconsists of the core classes for HDR imaging.
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import enum, rawpy, colour, imageio, copy, os, functools, skimage.transform
import numpy as np
from . import utils, processing, metadata
import preferences.preferences as pref

imageio.plugins.freeimage.download()

# -----------------------------------------------------------------------------
# --- Class imageType --------------------------------------------------------
# -----------------------------------------------------------------------------
class imageType(enum.Enum):
    """
    This class contains a list of constant values for different type of images.
    
    The constants are : SDR, RAW, HDR.
    """

    SDR = 0 # SDR image:                (.jpg)
    ARW = 1 # raw image file: sony ARW  (.arw)
    HDR = 2 # hdr file:                 (.hdr)

# -----------------------------------------------------------------------------
# --- Class channel ----------------------------------------------------------
# -----------------------------------------------------------------------------
class channel(enum.Enum):
    """
    This class groups constant values and methods relative to a channel of an image together. This class is an easy definition of channel.
    
    Attributes :
        value: int
            Store the type of channel.
            
    """
    
    sR      = 0
    sG      = 1
    sB      = 2

    sRGB    = 3

    X       = 4
    Y       = 5
    Z       = 6

    XYZ     = 7

    L       = 8
    a       = 9
    b       = 10

    Lab     = 11

    def colorSpace(self):
        """
        Retrieve the channel as a string.
        
        Returns:
            str
                Colorspace type of the channel : 'sRGB', 'XYZ' or 'Lab'.
        """
        csIdx = self.value // 4
        res = None
        if csIdx   == 0:    res ='sRGB'
        elif csIdx == 1:    res = 'XYZ'
        elif csIdx == 2:    res = 'Lab'

        return res

    def getValue(self):
        """
        Retrieve the channel index.
        
        Returns:
            int
                Index of the channel.
        """
        return self.value % 4

    @staticmethod
    def toChannel(s):
        """
        Convert a channel name as a constant value defined in this class.
        
        Args:
            s: str
                Name of the desired channel. This argument have to be in the list : 'sR', 'sG', 'sB', 'X', 'Y', 'Z', 'L', 'a' or 'b'.
            
        Returns:
            int
                Index of the channel.
        """
        if s=='sR' :    return channel.sR
        elif s=='sG' :  return channel.sG
        elif s=='sB' :  return channel.sB

        elif s=='X' :   return channel.X
        elif s=='Y' :   return channel.Y
        elif s=='Z' :   return channel.Z

        elif s=='L' :   return channel.L
        elif s=='a' :   return channel.a
        elif s=='b' :   return channel.b

        else:           return channel.L       
# -----------------------------------------------------------------------------        
# --- Class Image ------------------------------------------------------------
# -----------------------------------------------------------------------------
class Image(object):
    """
    class hdrCore.image.Image: the basic class that models HDR image and associated data
    
    Attributes:
        path (str):                 path to access image
        name (str):                 image filename
        colorData (numpy.ndarray):  numpy array of pixels
        type (image.imageType):     type of image
        linear (boolean):           image encoding is linear
        colorSpace (colour.models.RGB_COLOURSPACES): colorspace
        scalingFactor (float):      scaling factor to range[0..1]
        metadata (hdrCore.metadatametadata): metadata
        histogram(hdrCore.image.Histogram): image histogram   
    
    Methods:
        isHDR:                      (boolean) returns True if image is HDR
        process:                    (hdrCore.image.Image) computes a processing and returns a new Image 
        write:                      () write image and json metadata on disk (HDR image only)
        getChannel:                 ()
        getDynamicRange:
        buildHistogram:
        plot:
        __repr__:                   (str)
        split:                      (list[[hdrCore.image.Image]]) split an image into sub-images

    Static methods:
       read:                        (hdrCore.image.Image) read an image from file
       toOne:                       /!\ not used in uHDR
       buildLchColorData:
    """

    def __init__(self, path, name, colorData, type, linear, colorspace, scalingFactor=1.0):
        """__init__(): constrctor of class image.Image
            
        Args:
            path(str):              path to iamge file.
            name(str):              image filename.
            colorData(numpy.ndarray): array of pixels
            type(hdrCore.image.imageType): image type.
            linear(boolean):        True is image is boolean.
                after reading image from file:  .hdr image is linear
                                                .jpg image is not linear
            colorspace(colour.models.RGB_COLOURSPACES): image color space   
            scalingFactor(float, optional): scaling to factor to range [0,1] 
                /!\ not used in uHDR
        """

        self.path           = path                          # path to file          (str)
        self.name           = name                          # filename              (str)
        self.colorData      = colorData                     # float color data      (numpy.ndarray)
        self.shape          = self.colorData.shape          # image size height,width, color channel number (tuple)
        self.type           = type                          # image type            (hdrCore.image.imageType)
        self.linear         = linear                        # image in RGB linear   (bool)
        self.colorSpace     = colorspace                    # colorSpace            (colour.models.RGB_COLOURSPACES)
        self.scalingFactor  = scalingFactor                 # scaling to factor to range [0,1] (float)
        self.metadata       = None                          # associated meta data  (hdrCore.metadata.metadata)   
        self.histogram      = None                          # histogram             (hdrCore.image.Histogram)

    def isHDR(self):
        """isHDR: return True is image is HDR
        
        Args:

        Returns:
            (boolean)
                True if the image is an HDR image, False otherwise
        """
        return self.type == imageType.HDR

    def process(self, process, **kwargs):
        """
        compute a process according to the Processing object parameter.
        
        process, instance of Processing class, has a compute(image, \*\*kwargs) method
        
        Args:
            process: Processing object (see processing.Processing)
                
            kwargs: optionnal parameters packed as dict
                parameters of the compute method of the Processing object
        """
        return process.compute(self,**kwargs)

    @staticmethod
    def read(filename, thumb = False):
        """
        Method to read image from its filename. blablabla TODO à compléter.
        
        Args:
            filename: str
                Filename of the file to read.
            thumb: boolean
                This flag indicates us if soft have to read the thumbnail or the original file for what the thumbnail will be created.
                
        Returns:
            image.Image
                the image object build from file.

        Example:
        
        >>> import uhdCore
        >>> img = Image.read(filename = "test.hdr", thumb = True)
        
        TODO - Exemple à modifier
        """

        imgDouble, imgDoubleFull = None, None
        # image name
        path, name, ext = utils.filenamesplit(filename)

        # load raw file using rawpy
        if ext=="arw":
            outBit = 16
            raw = rawpy.imread(filename)
            ppParams = rawpy.Params(demosaic_algorithm=None, half_size=False, 
                                            four_color_rgb=False, dcb_iterations=0, 
                                            dcb_enhance=False, fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off, 
                                            noise_thr=None, median_filter_passes=0, 
                                            use_camera_wb=True,                                 # default False
                                            use_auto_wb=False, 
                                            user_wb=None, 
                                            output_color=rawpy.ColorSpace.sRGB,                 # output in SRGB
                                            output_bps=outBit,                                  # default 8
                                            user_flip=None, user_black=None, 
                                            user_sat=None, no_auto_bright=False, 
                                            auto_bright_thr=None, adjust_maximum_thr=0.75, 
                                            bright=1.0, highlight_mode=rawpy.HighlightMode.Clip, 
                                            exp_shift=None, exp_preserve_highlights=0.0, 
                                            no_auto_scale=False,
                                            gamma=None,                                         # linear output
                                            chromatic_aberration=None, bad_pixels_path=None)
            imgDouble = colour.utilities.as_float_array(raw.postprocess(ppParams))/(pow(2,16)-1)
            raw.close()
            scalingFactor, type, linear = 1.0, imageType.ARW, True

        # load jpg, tiff, hdr file using colour
        elif ext=="jpg":
            imgDouble = colour.read_image(filename, bit_depth='float32', method='Imageio')
            scalingFactor, type, linear = 1.0, imageType.SDR, False

        # post processing for HDR scaling to [ ,1]
        elif ext =="hdr":
            if thumb: 
                # do not read input only the thumbnail
                searchStr = os.path.join(path,"thumbnails","_"+name+"."+ext)
                if os.path.exists(searchStr): 
                    imgDouble = colour.read_image(searchStr, bit_depth='float32', method='Imageio') # <--- read thumbnail of input file

                else:
                    if not os.path.exists(os.path.join(path,"thumbnails")): os.mkdir(os.path.join(path,"thumbnails"))

                    # read image and create thumbnail
                    imgDouble = colour.read_image(filename, bit_depth='float32', method='Imageio') # <--- read input file

                    # resize to thumbnail size
                    iY, iX, _ = imgDouble.shape
                    maxX = processing.ProcessPipe.maxSize
                    factor = maxX/iX
                    imgDoubleFull = copy.deepcopy(imgDouble)
                    imgThumbnail =  skimage.transform.resize(imgDouble, (int(iY * factor),maxX ))
                    # save thumbnail
                    colour.write_image(imgThumbnail,searchStr, method='Imageio')

                    imgDouble = imgThumbnail

            else:
                # thumb set to False, read input not the thumbnail
                imgDouble = colour.read_image(filename, bit_depth='float32', method='Imageio')

            type = imageType.HDR
            linear = True
            scalingFactor = 1.0

        # create image object
        res =  Image(path, name+'.'+ext, np.float32(imgDouble),type, linear, None, scalingFactor)           # colorspace = None will be set in metadata.metadata.build(res)
        res.metadata = metadata.metadata.build(res)                                                         # build metadata (read if json file exists, else recover from exif data)

        # update path
        res.metadata.metadata['path'] = copy.deepcopy(path)
        # update size
        if thumb and isinstance(imgDoubleFull, np.ndarray):
            h,w, c = imgDoubleFull.shape
            res.metadata.metadata['exif']['Image Width']    = w
            res.metadata.metadata['exif']['Image Height']   = h

        # update image.colorSpace from metadata
        RGBcolorspace = ColorSpace.sRGB() # delfault color space
        if not ('sRGB' in res.metadata.metadata['exif']['Color Space']):
            res.colorSpace = RGBcolorspace 
        else: # 'sRGB' in color space <from exif>
            res.colorSpace = RGBcolorspace 

        # display ready image
        if 'display' in res.metadata.metadata.keys():
            disp = res.metadata.metadata['display']
            if disp in pref.getHDRdisplays().keys():
                scaling = pref.getHDRdisplays()[disp]['scaling']
                res.colorData = res.colorData/scaling  

        return res

    def write(self,filename):
        """
        Method to write the image in a file.

        Args:
            filename: str
                Filename of the file to read.
        """
        if self.isHDR():

            path, name, ext = utils.filenamesplit(filename)
            colour.write_image(self.colorData,filename, method='Imageio')

            # update filename related metadata before saving
            self.name = name+'.'+ext
            self.path  = path
            self.metadata.image = self
            self.metadata.metadata['filename'] = name+'.'+ext
            self.metadata.metadata['path'] =     path

            self.metadata.save()

    @staticmethod
    def toOne(colorData):
        """
        This method scale the image colorData in [0, 1] space

        Args:
            colorData: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """

        imgVector = utils.ndarray2vector(colorData)
        R, G, B = imgVector[:,0], imgVector[:,1], imgVector[:,2]
        maxRGB = max([np.amax(R), np.amax(G), np.amax(B)]) 

        return colorData/maxRGB, 1.0/maxRGB

    def getChannel(self,channel):
        """
        To get channel : works only for sR|sG|sB, X|Y|Z and L|a|b

        Args:
            channel: TODO
                TODO
        """

        # take into account colorSpace
        destColor = channel.colorSpace()
        image = processing.ColorSpaceTransform().compute(self,dest=destColor)

        if channel.getValue() <3:
            return image.colorData[:,:,channel.getValue()]
        else:
            return None

    def getDynamicRange(self,percentile=None):
        """
        Retrieve the dynamic range of image

        Args:
            percentile: float
                Optional  : percentile if None: just remove zero values
                
        Returns:
            float
                The dynamic range of the image
        """

        Y_min,Y_max = None, None
        Y = self.getChannel(channel.Y)

        if percentile == None : Y_min, Y_max = np.amin(Y[Y>0]), np.amax(Y)                  # use min and max
        else: Y_min, Y_max = np.percentile(Y[Y>0],percentile), np.percentile(Y,100-percentile)   # percentile

        return np.log2(Y_max)-np.log2(Y_min)

    #def getMinMaxPerChannel(self):
    #    """TODO - documentation de la méthode getMinMaxPerChannel
    #    """

    #    img = self.colorData
    #    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    #    minR, minG, minB = np.amin(R), np.amin(G), np.amin(B)
    #    maxR, maxG, maxB = np.amax(R), np.amax(G), np.amax(B)

    #    return ((minR,maxR),(minG,maxG),(minB,maxB))
    
    def buildHistogram(self,channel):
        """
        Build the histogram of the image

        Args:
            channel: TODO
                TODO
        """
        self.histogram = Histogram.buildbuild(self, channel, nbBins=100, range= None, logSpace = self.isHDR())

    def plot(self,ax,displayTitle=False,title=None,forceToneMapping=True,TMO=None):
        """
	Method to plot the image.

        Args:
            ax: TODO
                TODO
            displayTitle: boolean
                Indicates if the name and colorSpace must be displayed above the image.
            title: str
                Title for the image. If None, the title will be the name of the image and its colorspace.
            forceToneMapping: boolean
                TODO
            TMO: TODO
                TODO
        """
        # default values
        if not title: title= self.name+"("+self.colorSpace.name +" | "+ str(self.type)+")"
        if not TMO: TMO = processing.tmo_cctf()
        if (not (self.type == imageType.HDR)) or (not  forceToneMapping): 
            ax.imshow(self.colorData)
        else: # HDR and forceToneMapping
            ax.imshow(self.process(TMO).process(processing.clip()).colorData)
            title = title+"[auto TMo]"
        if displayTitle:  ax.set_title(title)
        ax.axis("off")

    def __repr__(self):
        """
        Converts an image as a string. This method dive all the informations of the image.
        """
        if not self.colorSpace:
            colorSpaceSTR = "None"
        else: 
            colorSpaceSTR =  self.colorSpace.name
        res =   "<class Image:\n" + \
                "\t name: " + self.name + "\n" + \
                "\t path: " + self.path + "\n" + \
                "\t shape: " + str(self.shape) +  "\n" + \
                "\t type: " + str(self.type) +  "\n" +\
                "\t linear: " + str(self.linear) +  "\n" +\
                "\t scaling to [0,1]: " + str(self.scalingFactor)+"\n" +\
                "\t color space: " + colorSpaceSTR + ">"
        return res

    @staticmethod
    def buildLchColorData(L,c,h,size,width,height):
        """
        TODO - Documentation de la méthode buildLchColorData

        Args:
            L: TODO
                TODO
            c: TODO
                TODO
            h: TODO
                TODO
            size: TODO
                TODO
            width: TODO
                TODO
            height: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """

        colorData = np.zeros((size[0], size[1],3))
        Lmin, Lmax = L if L[0] < L[1] else (L[1], L[0])
        cmin, cmax = c if c[0] < c[1] else (c[1], c[0])
        hmin, hmax = h

        xmin, xmax = 0, size[1]
        ymin, ymax = 0, size[0]

        if width=='L':
            if height=='c':
                #     +-------------------------------+
                #  c  |                               |
                #     +-------------------------------+
                #                Lightness
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        Lu = Lmin*(1-u)+Lmax*u
                        cv = cmin*(1-v) + cmax*v
                        hh = (hmin+hmax)/2
                        colorData[y,x,:] = [Lu,cv,hh]
            elif height=='h':
                #     +-------------------------------+
                #  h  |                               |
                #     +-------------------------------+
                #                Lightness
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        Lu = Lmin*(1-u)+Lmax*u
                        if hmin <= hmax:
                            hv = hmin*(1-v) + hmax*v
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hv = (hmin-360)*(1-v) + hmax*v
                            if hv < 0: hv = 360 +hv
                        cc = (cmin+cmax)/2
                        colorData[y,x,:] = [Lu,cc,hv]
        elif width=='c':
            if height=='L':
                #     +-------------------------------+
                #  L  |                               |
                #     +-------------------------------+
                #                chroma
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        cu = cmin*(1-u) + cmax*u
                        Lv = Lmin*(1-v)+Lmax*v
                        hh = (hmin+hmax)/2
                        colorData[y,x,:] = [Lv,cu,hh]
            elif height=='h':
                #     +-------------------------------+
                #  h  |                               |
                #     +-------------------------------+
                #                chroma
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        cu = cmin*(1-u) + cmax*u
                        if hmin <= hmax:
                            hv = hmin*(1-v) + hmax*v
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hv = (hmin-360)*(1-v) + hmax*v
                            if hv < 0: hv = 360 +hv
                        LL = (Lmin+Lmax)/2
                        colorData[y,x,:] = [LL,cu,hv]
        elif width == 'h':
            if height=='L':
                #     +-------------------------------+
                #  L  |                               |
                #     +-------------------------------+
                #                hue
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        if hmin <= hmax:
                            hu = hmin*(1-u) + hmax*u
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hu = (hmin-360)*(1-u) + hmax*u
                            if hu < 0: hv = 360 +hu
                        Lv = Lmin*(1-v)+Lmax*v
                        cc = (cmin+cmax)/2
                        colorData[y,x,:] = [Lv,cc,hu]
            elif height=='c':
                #     +-------------------------------+
                #  c  |                               |
                #     +-------------------------------+
                #                hue
                for x in range(xmin,xmax):
                    for y in range(ymin,ymax):
                        u, v = x/(xmax-1), y/(ymax-1)
                        if hmin <= hmax:
                            hu = hmin*(1-u) + hmax*u
                        else:
                            # hmin = 340 / hmax=20 -> hmin = hmin-360 >> hmin = -20, hmax =20
                            hu = (hmin-360)*(1-u) + hmax*u
                            if hu < 0: hv = 360 +hu
                        cv = cmin*(1-v)+cmax*v
                        LL = (Lmin+Lmax)/2
                        colorData[y,x,:] = [LL,cv,hu]

        return colorData

    def split(self,widthSegment,heightSegment):
        """split: split an image widthSegment x heightSegment sub-images.

            Args:
                widthSegment (int, Required): number of horizonatal segments
                heightSegment (int required): number of vertical segments

            Returns
                list[[hdrCore.image.Image]] : list of sub-images
        """
        imageHeight,imageWidth, _ = self.colorData.shape
        widthLimit = [(i*(imageWidth//widthSegment))  for i in range(widthSegment)]+[imageWidth]
        heightLimit = [(i*(imageHeight//heightSegment))  for i in range(widthSegment)]+[imageHeight]

        res = []

        for line in range(heightSegment):
            lines = []
            for col in range(widthSegment):
                cData =  copy.deepcopy(self.colorData[(heightLimit[line]):(heightLimit[line+1]),(widthLimit[col]):(widthLimit[col+1]),:])
                imgTemp = Image(copy.deepcopy(self.path), copy.deepcopy(self.name), cData, copy.deepcopy(self.type), self.linear, copy.deepcopy(self.colorSpace), self.scalingFactor)
                imgTemp.metadata = copy.deepcopy(self.metadata)
                lines.append(imgTemp)
            res.append(lines)

        return res

    @staticmethod
    def merge(imgList):
        """merge: merge 2D list of images into a single image (same size, same characteristics)

            Args:
                imgList (list[[hdrCore.image.Image]], Required): 2D list of images

            Returns:
                (hdrCore.image.Image)
        """
        totalWidth= functools.reduce(lambda x,y: x+y, map(lambda img: img.colorData.shape[1],imgList[0]),0)
        totalHeight= functools.reduce(lambda x,y: x+y,map(lambda imgList: imgList[0].shape[0],imgList),0)

        cData = np.zeros((totalHeight,totalWidth,3))

        y = 0
        for line in imgList:
            x=0
            for img in line:
                cData[y:(y+img.colorData.shape[0]),x:(x+img.colorData.shape[1]),:] = img.colorData
                x = x+img.colorData.shape[1]
            y = y + line[-1].colorData.shape[0]
        return Image(copy.deepcopy(imgList[0][0].path), copy.deepcopy(imgList[0][0].name), cData, copy.deepcopy(imgList[0][0].type), imgList[0][0].linear, copy.deepcopy(imgList[0][0].colorSpace), imgList[0][0].scalingFactor)
# -----------------------------------------------------------------------------
# --- Class ColorSpace -------------------------------------------------------
# -----------------------------------------------------------------------------
class ColorSpace(object):
    """
    Mapping to colour.models.RGB_COLOURSPACES
    """

    @staticmethod
    def Lab():
        """
        TODO - Documentation de la méthode Lab
        """
        return colour.RGB_Colourspace('Lab', primaries=np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]), whitepoint=np.array([0.32168, 0.33767]))
     
    @staticmethod                                 
    def Lch():
        """
        TODO - Documentation de la méthode Lch
        """
        return colour.RGB_Colourspace('Lch', primaries=np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]), whitepoint=np.array([0.32168, 0.33767]))

    @staticmethod
    def sRGB():
        """
        TODO - Documentation de la méthode sRGB
        """
        return colour.models.RGB_COLOURSPACES['sRGB'].copy()

    @staticmethod
    def scRGB():
        """
        TODO - Documentation de la méthode scRGB
        """
        colorSpace = colour.models.RGB_COLOURSPACES['sRGB'].copy()
        colorSpace.cctf_decoding=None
        return colorSpace

    @staticmethod
    def XYZ():
        """
        TODO - Documentation de la méthode XYZ
        """
        return colour.RGB_Colourspace('XYZ', primaries=np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]), whitepoint=np.array([0.32168, 0.33767]))

    @staticmethod
    def build(name='sRGB'):
        """
        TODO - Documentation de la méthode build

        Args:
            name='sRGB': str
                TODO
        """
        cs  = None
        if name== 'sRGB': cs =  ColorSpace.sRGB()
        if name== 'scRGB': cs =  ColorSpace.scRGB()
        if name== 'Lab' : cs =  ColorSpace.Lab()
        if name== 'Lch' : cs =  ColorSpace.Lch()
        if name== 'XYZ' : cs =  ColorSpace.XYZ()
        return cs 
# -----------------------------------------------------------------------------     
# --- Class Histogram --------------------------------------------------------
# -----------------------------------------------------------------------------
class Histogram(object):
    """
    TODO - documentation de la classe image.Histogram
    
    description of class
    
    Attributes:
        name: str
            TODO
        channel: TODO
            TODO
        histValue: TODO
            TODO
        edgeValue: TODO
            TODO
        logSpace: TODO
            TODO
    """
    
    def __init__(self,histValue,edgeValue,name,channel,logSpace=False):
        """
        TODO - Documentation de la méthode __init__
        
        /!\ - Les constructeurs n'apparaissent pas dans la doc générées par sphinx.

        Args:
            histValue: TODO
                TODO
            edgeValue: TODO
                TODO
            name: TODO
                TODO
            channel: TODO
                TODO
            logSpace: TODO
                TODO
        """
        self.name           = name
        self.channel        = channel
        self.histValue      = histValue
        self.edgeValue      = edgeValue
        self.logSpace       = logSpace    

    def __repr__(self):
        """
        TODO - Documentation de la méthode __repr__
        """
        res =   "<class Histogram: \n" + \
                "t name:"               + self.name                 + "\n"  + \
                "\t nb bins: "        + str(len(self.histValue))    + "\n"  + \
                "\t channel: "        + str(self.channel.name)      + "\n"  + \
                "\t logSpace: "       + str(self.logSpace)          + ">"  
        return res

    def __str__(self):
        """
        TODO - Documentation de la méthode __str__
        """
        return self.__repr__()

    def normalise(self,norm=None):
        """
        Normalise histogram according to norm='probability' | 'dot'

        Args:
            norm: str
                Norm of the normalization
                
        Returns:
            TODO
                TODO
        """
        res = copy.deepcopy(self)
        if not norm: norm = 'probability'
        if norm == 'probability':
            sum = np.sum(res.histValue)
            res.histValue = res.histValue/sum
        elif norm == 'dot':
            dot2 = np.dot(res.histValue,res.histValue)
            res.histValue = res.histValue/np.sqrt(dot2)
        else:
            print("WARNING[Histogram.normalise(",self.name,"): unknown norm:", norm,"!]")
        return res

    @staticmethod
    def build(img,channel,nbBins=100,range=None,logSpace=None):
        """
        Build an Histogram object from image.

        Args:
            img: Image
                Required : input image from witch hsitogram will be build
            channel: channel
                Required : image channel used to build histogram
            nbBins: int
                Optional : histogram number of bins
            range: (Float,Float)
                Optional : range of histogram, if None min max of channel
            logSpace: boolean
                Optional : compute in log space if True, if None guess from image
                
        Returns:
            Histogram
                Histogram of the image.
        """
        # logSpace
        if not isinstance(logSpace,(bool, str)): logSpace = 'auto'
        if isinstance(logSpace,str):
            if logSpace=='auto':
                if img.type == image.imageType.imageType.SDR : logSpace = False
                if img.type == image.imageType.imageType.HDR : logSpace = True
            else: logSpace = False

        channelVector = utils.ndarray2vector(img.getChannel(channel))
        # range
        if not range: 
            if channel.colorSpace() == 'Lab':
                range= (0.0,100.0)
            elif channel.colorSpace() == 'sRGB'or channel.colorSpace() == 'XYZ':
                range= (0.0,1.0)
            else:
                range= (0.0,1.0)

        # compute bins
        if logSpace:
            minChannel = np.amin(channelVector)
            maxChannel = np.amax(channelVector)
            #bins
            bins = 10 ** np.linspace(np.log10(minChannel), np.log10(maxChannel), nbBins+1)
        else:
            bins = np.linspace(range[0],range[1],nbBins+1)

        nphist, npedges = np.histogram(channelVector, bins)

        nphist = nphist/channelVector.shape
        return Histogram(nphist, 
                         npedges, 
                         'hist_'+str(channel)+'_'+img.name, 
                         channel,
                         logSpace = logSpace
                         )

    def plot(self,ax,color='r',shortName=True,title=True):
        """
        Method to plot the histogram

        Args:
            ax: TODO
                TODO
            color: TODO
                TODO
            shortName: boolean
                TODO
            title: boolean
                TODO
        """
        if not color : color = 'r'
        ax.plot(self.edgeValue[1:],self.histValue,color)
        if self.logSpace: ax.set_xscale("log")
        name = self.name.split("/")[-1]+"(H("+self.channel.name+"))"if shortName else self.name+"(Histogram:"+self.channel.name+")"
        if title: ax.set_title(name)

    def toNumpy(self):
        """
        TODO - Documentation de la méthode toNumpy
        
        Returns:
            TODO
                TODO
        """
        return self.histValue
