# -*- coding: utf-8 -*-

"""
    This package is a set of utilities for working with the Leap Motion Sensor
    Frame data. Note that the code in the functions were taken from the Leap
    Motion API documentation.

__author__  : "Stephen Terry"
__date__    : "09/16/17"
__version__ : "Version 0.1"
"""

import array
import cv2
import ctypes
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import struct
import sys
import timeit

# Paths to Leap SDK
src_dir = r'S:\LeapDeveloperKit_3.2.0+45899_win\LeapDeveloperKit_3.2.0+45899_win\LeapSDK\lib'
arch_dir = r'S:\LeapDeveloperKit_3.2.0+45899_win\LeapDeveloperKit_3.2.0+45899_win\LeapSDK\lib\x64'

# Leap SDK must be inserted in system path
sys.path.insert(0,src_dir)
sys.path.insert(0,arch_dir)
# import the API
import Leap

def handMoving(hand):
    """This function will check whether the hand under test is moving or not.  A
       threshold value is used to check for movement, since the Leap sensor reports
       movement in millimeters.
       
       PARAMETERS:   hand: Leap.Hand
                       The hand to check for movement
                       
       RETURNS:      moving: boolean
                       True if hand is moving"""
    
    moving = False
    x = hand.palm_velocity.x
    y = hand.palm_velocity.y
    z = hand.palm_velocity.z
    
    SPEEDTHRESH = 10
    
    handSpd = np.sqrt(x**2 + y**2 + z**2)
    
    #print "HANDMOVING",handSpd
    
    if handSpd > SPEEDTHRESH:
        #print "HandMove(): Hand Moved"
        moving = True
    
    else:
        #print "HandMove(): %s" % handSpd
        ## Hand is considered not moving. Check fingers
        for i in range(0,5):
            fngr = hand.fingers[i]
            fx = fngr.tip_velocity.x
            fy = fngr.tip_velocity.y
            fz = fngr.tip_velocity.z
            fngrSpd = np.sqrt(fx**2 + fy**2 + fz**2)
            #print "FINGERMOVING",fngrSpd
            if fngrSpd > (SPEEDTHRESH):
                #print "HandMove(): Finger Moved"
                moving = True
                break
    
    return moving

def handChanged(pHand,cHand):
    
    """This function will check the distance between a point in two frames (hands).
       If the distance between the points exceeds the threshold, the hand is 
       considered to be moving.
       
       PARAMETERS:   pHand,cHand: Leap.Hand
                       the previouis hand and the current hand to check for 
                       movement
                       
       RETURNS:     changed: boolean
                       True if hand has changed position"""
    
    changed = False
    MOVETHRESH = 7
    
    px = pHand.palm_position.x
    py = pHand.palm_position.y
    pz = pHand.palm_position.z
    cx = pHand.palm_position.x
    cy = pHand.palm_position.y
    cz = pHand.palm_position.z
    
    #print "HANDCHANGE",distanceR3(px,cx,py,cy,pz,cz)
    #print pHand.frame.id,cHand.frame.id
    #print "Valid: %s,%s" %(pHand.is_valid,cHand.is_valid)
    if distanceR3(px,cx,py,cy,pz,cz) > MOVETHRESH:
        #print "HANDCHANGE(): Hand Moved"
        changed = True
        
    else:
        ## Hand is considered to not be changed.  Check fingers
        for i in range(0,5):
            pfngr = pHand.fingers[i]
            cfngr = cHand.fingers[i]
            px = pfngr.tip_position.x
            py = pfngr.tip_position.y
            pz = pfngr.tip_position.z
            cx = cfngr.tip_position.x
            cy = cfngr.tip_position.y
            cz = cfngr.tip_position.z
            #print "FINGERCHANGE",distanceR3(px,cx,py,cy,pz,cz)
            if distanceR3(px,cx,py,cy,pz,cz) > MOVETHRESH:
                #print "HANDCHANGE(): Finger Moved"
                changed = True
                break

        
    return changed
    

def distanceR3(x1,x2,y1,y2,z1,z2):
    """This function returns the Euclidean distance between points in a 3D space.
    
       PARAMETERS:  xn,yn,zn : integer or float
                     the xyz values for the endpoints
                     
       RETURNS:  distance : integer or float"""
    
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) 

def putHandInIBox(palmPos,iBox):
    
    """This function will check to see if a Leap Hand object is within the bounds
       of the Leap Frame Interaction Box.  Will return a message indicating if 
       hand is in box, or how to get hand in box.
       
       PARAMETERS:   palmPos: Leap.Vector
                       A Leap.Vector containing the X/Y/Z position of the palm
                     
                     iBox:  Leap.InteractionBox
                       The Leap.Interaction box to compare position to.
                       
       RETURNS:    msg: str
                     A string containing the directions on how to place hand in
                     the InteractionBox.
    """
    
    iBoxWidth = iBox.width # x-axis
    iBoxHeight = iBox.height # y-axis
    iBoxDepth = iBox.depth # z-axis
    iBoxCenter = iBox.center # vector
    msg = ""
    ## Assuming orientation is facing user,light down (bottom right corner),plug on left
    if palmPos.z > (iBoxCenter.z + (iBoxDepth / 2)):
        msg = "Move hand up"
    if palmPos.z < (iBoxCenter.z - (iBoxDepth / 2)):
        msg = "Move hand down"
    if palmPos.x > (iBoxCenter.x + (iBoxWidth / 2)):
        if msg == "":
            msg = "Move hand left"
        else:
            msg = msg + ", left"
    if palmPos.x < (iBoxCenter.x - (iBoxWidth / 2)):
        if msg == "":
            msg = "Move hand right"
        else:
            msg = msg + ", right"
    if palmPos.y > (iBoxCenter.y + (iBoxHeight / 2)):
        if msg == "":
            msg = "Move hand forward"
        else:
            msg = msg + ", forward"
    if palmPos.y < (iBoxCenter.y - (iBoxHeight / 2)):
        if msg == "":
            msg = "Move hand backward"
        else:
            msg = msg + ", backward"
    if msg == "":
        msg = "Hand in position"
    
    return msg

def image_to_np_array(image):
    
    """This function takes a Leap Image object and converts it to a numpy array.
    
       PARAMETERS:  image: Leap.Image
                       A Leap.Image object to be converted to numpy array
                       
       RETURNS  :   img: numpy.ndarray
                       The numpy array of the image data"""
                       
    if image.is_valid:
        i_address = int(image.data_pointer)
        ctype_array_def = ctypes.c_ubyte * image.height * image.width
        # as ctype array
        as_ctype_array = ctype_array_def.from_address(i_address)
        # as numpy array
        as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
        img = np.reshape(as_numpy_array, (image.height, image.width))
        
        return img
    
    else:
        sys.stdout.write("Image is not valid\n")
        return None

def images_to_np_arrays(images):
    
    """This funtion takes a Leap ImageList object and converts the images in the
       imagelist to np arrays.
       
       PARAMETERS:  images: Leap.ImageList
                    the imagelist object from the Leap.Frame object
                    
       RETURNS:    l_img,r_img: numpy.ndarray
                   the numpy arrays of the imagelist data"""
                   
    # In the image list, the left image is the first image
    l_img = image_to_np_array(images[0])
    r_img = image_to_np_array(images[1])
      
    return l_img,r_img

def saveImages(frame,count,path):
    
    """This function will write the Leap images to the specified path.
    
       PARAMETERS:   frame: Leap.Frame
                           the frame containing the images
                           
                     count: integer
                           the count of the image taken
                           
                     path:  string
                           the output path for the image
                           
       RETURNS:  NONE"""
    
    images = frame.images
    #wrap image data in numpy array
    if images[0].is_valid and images[1].is_valid:
        li_address, ri_address = int(images[0].data_pointer), int(images[1].data_pointer)
        ctype_array_def_l, ctype_array_def_r = ctypes.c_ubyte * images[0].height * images[0].width,ctypes.c_ubyte * images[1].height * images[1].width
        # as ctypes array
        as_ctype_array_l = ctype_array_def_l.from_address(li_address)
        as_ctype_array_r = ctype_array_def_r.from_address(ri_address)
        # as numpy array
        as_numpy_array_l = np.ctypeslib.as_array(as_ctype_array_l)
        as_numpy_array_r = np.ctypeslib.as_array(as_ctype_array_r)
        limg = np.reshape(as_numpy_array_l, (images[0].height, images[0].width))
        rimg = np.reshape(as_numpy_array_r, (images[1].height, images[1].width))
    
        imagePath = path + "\\Images"
        lname = "left_image_%s.jpg" % str(count).zfill(5)
        rname = "right_image_%s.jpg" % str(count).zfill(5)
        scipy.misc.toimage(limg, cmin=np.min(limg), cmax=np.max(limg)).save(imagePath+"\\"+lname)
        scipy.misc.toimage(rimg, cmin=np.min(rimg), cmax=np.max(rimg)).save(imagePath+"\\"+rname)
    else:
        print images[0].is_valid, images[1].is_valid
        print images.is_empty

###############################################################################
# The following functions do not work as functions.  See the Leap Motion API IRT
# serialization/deserialization of frames.  The behaviour of serialization is 
# undefined when passing Leap frames through functions.  The functions are left
# here as a reference on how to read/write serialized data.
def serializeData(frame):
    
    """This function serializes the data from a leap motion frame to prepare
       it to be saved to file. Note that serialization of data does not keep
       the image data, so image data must be recorded separately if intended
       to keep.  Also note, that not all frames are the same size, so a separate
       function would need to be created to write multiple frames to one file.
       See Leap Motion API Serialization for more info.
       
       PARAMETERS :  frame: Leap.Frame
                       a single frame from the leap motion controller
                       
       RETURNS:    buffer: buffer
                       a serialized leap motion frame"""

    #serialName = "frame_%s.data" % str(count).zfill(5)
    #serialPath = path + "\\Serialized\\" + serialName
    
    serialized_tuple = frame.serialize
    serialized_data = serialized_tuple[0]
    serialized_length = serialized_tuple[1]
    data_address = serialized_data.cast().__long__()
    bfr = (ctypes.c_ubyte * serialized_length).from_address(data_address)
    #with open(serialPath, 'wb') as data_file:
    #    data_file.write(buffer)
    
    return bfr    

def deserializeData(fname, controller = None):
    
    """This function deserializes a serialized Leap Motion Frame that can then
       be treated as a Frame.  Note that images are not available in deserialized
       frames.  Note that a controller must be instantiated, but does not need 
       to be connected.
       
       PARAMETERS: fname: string
                   the full path to the serialized frame output
                   
                   controller: Leap.Controller
                   the controller required to deserialize data
                   
      RETURNS:     frame: Leap.Frame
                   the deserialized frame object"""    
    
    if controller is None:
        try:
            controller = Leap.Controller()
        except:
            sys.stdout.write("Error creating Leap.Controller object\n")
            return None

    # need an empty frame to populate    
    #frame = Leap.Frame()
    frame = controller.frame()

    with open(fname, 'rb') as data_file:
        data = data_file.read()

    leap_byte_array = Leap.byte_array(len(data))
    address = leap_byte_array.cast().__long__()
    ctypes.memmove(address, data, len(data))

    frame.deserialize((leap_byte_array, len(data)))

    return frame

###############################################################################
