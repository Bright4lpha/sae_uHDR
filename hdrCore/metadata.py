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
import enum, rawpy, colour, imageio, json, os, subprocess, ast, copy
import numpy as np
from . import utils, processing, image
import preferences.preferences as pref

# -----------------------------------------------------------------------------
# --- Class tags --------------------------------------------------------------
# -----------------------------------------------------------------------------
class tags:
    """
    the class tags is use to defines tags that can be set to an image. The tags definition are defined in a json file (./preferences/tags.json).
    tags are group in tags group (only one level of hierarchy !
    """
    def __init__(self):
        if os.path.exists("./preferences/tags.json"):
            with open('./preferences/tags.json') as f: self.tags =  json.load(f)
        else:
            self.tags = {'no-tag':[]}
    # ---------------------------------------------------------------------------
    def getTagsRootName(self):
        return list(self.tags.keys())[0]
    # ---------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# --- Class metadata ----------------------------------------------------------
# -----------------------------------------------------------------------------
class metadata:
    """
    The metadatas of the image can be store in an object of this class. The informations
    which can be stored are : filename, path, a description, the exif data, the category,
    the processes to apply on the image.
    
    Atributes:
        metadata: TODO
            TODO
        image: TODO
            TODO
    """

    # colorspace used if unknown oe undefined color space
    defaultColorSpaceName = 'sRGB'

    def __init__(self,_image):
        """
        TODO - Documentation de la méthode __init__
        
        /!\ - Les constructeurs n'apparaissent pas dans la doc générées par sphinx.

        Args:
            _image: TODO
                TODO
        """

        self.metadata =  {
            'filename': None,
            'path':     None,
            'description': None,
            'exif': {'Image Width': None,
                     'Image Height': None,
                     'Dynamic Range (stops)':None, 
                     'Exposure Time': None,
                     'F Number': None,
                     'ISO': None,
                     'Bits Per Sample': None,
                     'Color Space': None,
                     'Camera': None,
                     'Software': None,
                     'Lens': None,
                     'Focal Length': None},
            #'hdr-use-case': [
            #    {'inside': {
            #        'Window with view on bright outdoor': None,
            #        'High Contrast and illuminants tend to be over-exposed':None,
            #        'Backlit portrait':None}},
            #    {'outside':{
            #        'Sun in the frame':None,
            #        'Backlight':None,
            #        'Shadow and direct lighting':None,
            #        'Backlit portrait':None,
            #        'Nature':None,
            #        'Quite LDR':None}},
            #    {'lowlight':{
            #        'Portrait':None,
            #        'Outside with bright illuminants':None,
            #        'Cityscape':None,
            #        'Event':None}},
            #    {'special':{
            #        'Shiny object  and specular highlights':None,
            #        'Memory colors':None,
            #        'Scene with color checker or gray chart':None,
            #        'Translucent objects and stained glass':None,
            #        'Traditional tone mapping failing cases':None}}],
            'processpipe': None,
            'display' : None
            }
        # other metadta from tags.json
        self.otherTags = tags()
        self.metadata[self.otherTags.getTagsRootName()] = copy.deepcopy(self.otherTags.tags[self.otherTags.getTagsRootName()])

        self.metadata['filename'] = _image.name
        self.metadata['path'] =     _image.path
        h,w, c = _image.shape
        self.metadata['exif']['Image Width']    = w
        self.metadata['exif']['Image Height']   = h
        self.image = _image                             # reference to image (Image)    files: ./image.py
    
    @staticmethod
    def build(_image):
        """
        Build metadata object from an image.

        Args:
            _image: TODO
                TODO
                
        Returns:
            TODO
                TODO
        """

        res = metadata(_image)

        # check if metafile file exists
        splits = _image.name.split('.')
        filenameNoExt = '.'.join(splits[:-1])
        ext = splits[-1]
        filenameMetadata = filenameNoExt+'.json'
        JSONfilename = os.path.join(_image.path, filenameMetadata)   

        if os.path.isfile(JSONfilename):
            with open(JSONfilename, "r") as file:
                metaInFile = json.load(file)
                # copy all metadata from files
                if pref.keepAllMeta:
                    for keyInFile in metaInFile.keys():  self.metadata[keyInFile] = copy.deepcopy(metaInFile[keyInFile])
                else:
                    for keyInFile in metaInFile.keys():  
                        if keyInFile in res.metadata :
                            res.metadata[keyInFile] = copy.deepcopy(metaInFile[keyInFile])
                        else:
                            print(f'WARNING[metadata "{keyInFile}" not in "tags.json"  will be deleted! (consider changing "keepAllMeta" to "True" in  preferences.py)]')

                if _image.isHDR(): res.metadata['exif']['Color Space']=   'scRGB'

        else:
            exifDict = metadata.readExif(os.path.join(_image.path,_image.name))
            res.recoverData(exifDict)
            with open(JSONfilename, "w") as file: json.dump(res.metadata,file)

        return res
    # ---------------------------------------------------------------------------
    def save(self):
        """
        Save the metadata in a json file. The extension .json is added to the filename of the image.
        """
        filenameNoExt = '.'.join(self.image.name.split('.')[:-1])
        filenameMetadata = filenameNoExt+'.json'
        JSONfilename = os.path.join(self.image.path, filenameMetadata)   

        with open(JSONfilename, "w") as file: json.dump(self.metadata,file)
    # ---------------------------------------------------------------------------
    @staticmethod
    def readExif(filename):
        """
        Return exif (Dict) from image file

        Args:
            filename: str
                Name of the image file.
                
        Returns:
            TODO
                TODO
        """
        exifDict = dict()
        if os.path.isfile(filename): # check if filename exists
            if os.path.isfile('exiftool.exe'):# reading metadata with exiftool
                try:
                    exifdata = subprocess.check_output(['exiftool.exe','-a',filename],
                        shell=True, universal_newlines=True,
                        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    exifdata = exifdata.splitlines()
                except:
                    print("ERROR[metadata.readExif(",filename,"): error while reading!]")
                    exifdata =[]
                # buid exif dict
                for each in exifdata:
                    tag,val = each.split(':',1)# tags and values are separated by a semi colon
                    exifDict[tag.strip()] = val.strip()
            else: 
                print("ERROR[metadata.readExif(",filename,"): consider installing exiftool for better exif metadata, degraded mode with imageio!]")
                img = imageio.imread(filename)
                exifDict = img.meta['EXIF_MAIN'] if 'EXIF_MAIN' in img.meta else {}
        else: print("ERROR[metadata.readExif(",filename,"): file not found]")

        return exifDict
    # ---------------------------------------------------------------------------
    def recoverData(self,exif):
        """
        TODO - Documentation de la méthode recoverData

        Args:
            exif: TODO
                TODO
        """

        # data from exif
        if not (exif=={}): # exif not empty
            # colorSpace: 'Color Space', 'Profile Description', 'ColorSpace'
            exifColorSpace = ['Color Space', 'Profile Description', 'ColorSpace']
            if exifColorSpace[0] in exif: self.metadata['exif']['Color Space']= exif[exifColorSpace[0]]
            elif exifColorSpace[1] in exif: self.metadata['exif']['Color Space']= exif[exifColorSpace[1]]
            elif exifColorSpace[2] in exif:
                if exif[exifColorSpace[2]]==1:      self.metadata['exif']['Color Space']=   'sRGB'
                elif exif[exifColorSpace[2]]==2:    self.metadata['exif']['Color Space']=   'Adobe RGB (1998)'
                else:                               self.metadata['exif']['Color Space']=   metadata.defaultColorSpaceName
            else: 
                print("WARNING[",self.image.name, "exif does not contain 'Color Space' nor 'Profile Description', color space is set to sRGB  or scRGB for HDR images!]")
                self.metadata['exif']['Color Space'] = metadata.defaultColorSpaceName 

            # exposure time: 'Exposure Time', 'ExposureTime' 
            exifEposureTime = ['Exposure Time', 'ExposureTime']
            if exifEposureTime[0] in exif: 
                exposureTimeSTRs = exif[exifEposureTime[0]].split('/')
                numerator = ast.literal_eval(exposureTimeSTRs[0]) 
                denominator = ast.literal_eval(exposureTimeSTRs[1]) 
                self.metadata['exif']['Exposure Time']= (numerator,denominator)
            elif exifEposureTime[1] in exif: self.metadata['exif']['Exposure Time']= exif[exifEposureTime[1]]
            else: self.metadata['exif']['Exposure Time'] = None

            # exifAperture: 'F Number', 'FNumber' 
            exifAperture = ['F Number', 'FNumber']
            if exifAperture[0] in exif:
                apertureValue = exif[exifAperture[0]]
                if '/' in apertureValue:
                    apertureSTRs = apertureValue.split('/')
                    numerator = ast.literal_eval(apertureSTRs[0]) 
                    denominator = ast.literal_eval(apertureSTRs[1]) 
                    self.metadata['exif']['F Number']= (numerator,denominator)
                else: self.metadata['exif']['F Number']= (ast.literal_eval(apertureValue),1)
            elif exifEposureTime[1] in exif: self.metadata['exif']['F Number']= exif[exifAperture[1]]
            else: self.metadata['exif']['F Number'] = None
            
            # ISO: 'ISO', 'ISOSpeedRatings'
            exifISO = ['ISO', 'ISOSpeedRatings']
            if exifISO[0] in exif: self.metadata['exif']['ISO']= ast.literal_eval(exif[exifISO[0]])
            elif exifISO[1] in exif: self.metadata['exif']['ISO']= exif[exifISO[1]]
            else: self.metadata['exif']['ISO'] = None

            # Bits Per Sample: 'Bits Per Sample'
            exifBPS = ['Bits Per Sample']
            if exifBPS[0] in exif: self.metadata['exif']['Bits Per Sample']= ast.literal_eval(exif[exifBPS[0]])
            else: self.metadata['exif']['Bits Per Sample'] = None 

            # Camera: 'Make','Camera Model Name', 'Model'
            exifCamera = ['Make', 'Camera Model Name', 'Model']
            if exifCamera[0] in exif: 
                cameraMakeModel = exif[exifCamera[0]]
                if exifCamera[1] in exif:   cameraMakeModel += ' '+exif[exifCamera[1]]
                elif exifCamera[2] in exif: cameraMakeModel += ' '+exif[exifCamera[2]]
            else: cameraMakeModel = None
            self.metadata['exif']['Camera'] = cameraMakeModel

            # Software: 'Software'
            exifSoftware = ['Software']
            if exifSoftware[0] in exif: self.metadata['exif']['Software']= exif[exifSoftware[0]]
            else: self.metadata['exif']['Software'] = None

            # Lens: 'Lens Model', 'LensModel'
            exifLens = ['Lens Model', 'LensModel']
            if exifLens[0] in exif: self.metadata['exif']['Lens']= exif[exifLens[0]]
            elif exifLens[1] in exif: self.metadata['exif']['Lens']= exif[exifLens[1]]
            else: self.metadata['exif']['Lens'] = None

            # Focal Length: 'Focal Length', 'FocalLength'
            exifFocalLength = ['Focal Length', 'FocalLength']
            if exifFocalLength[0] in exif:
                focalLengthValue = exif[exifFocalLength[0]]
                focalLengthSTR = focalLengthValue.split(' ')[0]
                numerator = ast.literal_eval(focalLengthSTR) 
                self.metadata['exif']['Focal Length']= (numerator,1)
            elif exifFocalLength[1] in exif: self.metadata['exif']['Focal Length']= exif[exifFocalLength[1]]
            else: self.metadata['exif']['Focal Length'] = None
        
        else: # exif data = {}
            self.metadata['exif']['Color Space'] = metadata.defaultColorSpaceName
            print(" [META] >> metadata.recoverData(",self.image.name,").colorSpace undefined set it to:", 'sRGB or scRGB for HDR images')

        # bit per sample
        if self.metadata['exif']['Bits Per Sample'] == None:
            if self.image.name.endswith('.hdr'): self.metadata['exif']['Bits Per Sample'] =32
            if self.image.name.endswith('.jpg'): self.metadata['exif']['Bits Per Sample'] = 8

        # over data
        # dynamic range
        self.image.colorSpace= image.ColorSpace.sRGB()
        self.metadata['exif']['Dynamic Range (stops)'] = self.image.getDynamicRange(0.5)
    # ---------------------------------------------------------------------------
    def __repr__(self):
        """
        Convert the metadata in string.
        
        Returns:
            str
                The metadata in a string format.
        """
        return json.dumps(self.metadata)
    # ---------------------------------------------------------------------------
    def __str__(self):
        """
        Convert the metadata in string.
        
        Returns:
            str
                The metadata in a string format.
        """
        return self.__repr__()
# ---------------------------------------------------------------------------
