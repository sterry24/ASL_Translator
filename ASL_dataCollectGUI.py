# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 10:59:29 2017

@author: Stephen
"""

## Paths to Leap Motion SDK
src_dir = r'S:\LeapDeveloperKit_3.2.0+45899_win\LeapDeveloperKit_3.2.0+45899_win\LeapSDK\lib'
arch_dir = r'S:\LeapDeveloperKit_3.2.0+45899_win\LeapDeveloperKit_3.2.0+45899_win\LeapSDK\lib\x64'

import os
import sys
# Leap Motion SDK paths must be added to system path
sys.path.insert(0,src_dir)
sys.path.insert(0,arch_dir)
# add path to LEAPUTILS to sys path
# I don't think this is the correct way to handle this
sys.path.append(os.path.abspath('../'))

try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except ImportError:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *

import ctypes
import cv2
import math
import numpy as np
import pandas as pd
import scipy.misc
import string
import time

# import the Leap API
import Leap
# import leap utilities created for this project
import LEAPUTILS.leap_utilities as lutils

class FrameGrabber(QObject):
    
    """This class is the 'Worker' object for thread creating for generating
       the camera preview."""
    
    signalStatus = pyqtSignal(list)
    
    def __init__(self,parent=None):
        super(self.__class__,self).__init__(parent)
        
        self.controller = Leap.Controller()
        self.controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
        self.iBox = None
        ## Cannot instantiate QTimer here.  It must be instantiated in the thread
        ## in which it will be used.  So we will instantiate it when calling
        ## the startCamera() function from the QThread that controls it.
        
        
    @pyqtSlot()
    def startCamera(self):

        """This function instantiates a QTimer object that is used to call the
           runCamLoop() function at an interval of 1ms.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""        
        
        self._timer = QTimer()
        self._timer.timeout.connect(self.runCamLoop)
        self._timer.start(1)
    
    def runCamLoop(self):

        """This function grabs a frame from the Leap Motion Sensor.  The images
           are pulled from the frame, converted to numpy arrays, and then sent
           back to the main event loop via a PyQt signal emission.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""
        

        frame = self.controller.frame()
        if self.iBox is None:
            self.iBox = frame.interaction_box
#        self.iBoxWidth = self.iBox.width # x-axis
#        self.iBoxHeight = self.iBox.height # y-axis
#        self.iBoxDepth = self.iBox.depth # z-axis
#        self.iBoxCenter = self.iBox.center # vector
#        msg = ""
        if not frame.images.is_empty:
            images = frame.images
            if frame.hands.is_empty:
                msg = "No Hands in Frame"
            else:
                palmPos = frame.hands[0].palm_position
                msg = lutils.putHandInIBox(palmPos,self.iBox)
#                ## Assuming orientation is facing user,light down (bottom right corner),plug on left
#                if palmPos.z > (self.iBoxCenter.z + (self.iBoxDepth / 2)):
#                    msg = "Move hand up"
#                if palmPos.z < (self.iBoxCenter.z - (self.iBoxDepth / 2)):
#                    msg = "Move hand down"
#                if palmPos.x > (self.iBoxCenter.x + (self.iBoxWidth / 2)):
#                    if msg == "":
#                        msg = "Move hand left"
#                    else:
#                        msg = msg + ", left"
#                if palmPos.x < (self.iBoxCenter.x - (self.iBoxWidth / 2)):
#                    if msg == "":
#                        msg = "Move hand right"
#                    else:
#                        msg = msg + ", right"
#                if palmPos.y > (self.iBoxCenter.y + (self.iBoxHeight / 2)):
#                    if msg == "":
#                        msg = "Move hand forward"
#                    else:
#                        msg = msg + ", forward"
#                if palmPos.y < (self.iBoxCenter.y - (self.iBoxHeight / 2)):
#                    if msg == "":
#                        msg = "Move hand backward"
#                    else:
#                        msg = msg + ", backward"
#                if msg == "":
#                    msg = "Hand in position"
                
            limg,rimg = lutils.images_to_np_arrays(images)
            self.signalStatus.emit([limg,rimg,msg])

        else:
            #sys.stdout.write("Stopped Camera")                    
            self.signalStatus.emit([])
        
    @pyqtSlot()
    def stopCamera(self):

        """This function stops the QTimer object that is calling the runCamLoop
           function.  It then emits an empty signal that is used by the main event
           loop so that it knows the status of the thread.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""
        
        #sys.stdout.write("Stop Camera Called")
        self._timer.stop()
        self.signalStatus.emit([None])

class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow,self).__init__(parent)

        self.iBox = None
        self._dataPath = r"S:\DATA"
        self._dataFolders = ['Data','Images','Serialized']
        self._staticChars = list(string.ascii_uppercase)
        for i in range(9,0,-1):    # ASL number 10 is not static
            self._staticChars.insert(0,str(i))
        self._staticChars.remove('J')
        self._staticChars.remove('Z')
        self._numFramesToCollect = 50
        self._controller = Leap.Controller()
        self._controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
        self._camPreview = FrameGrabber()
        self._camPrevThread = QThread()
        
        self._camPreview.moveToThread(self._camPrevThread)
        self._camPrevThread.start()
        self._camPreview.signalStatus.connect(self.updateImage)
        
        self.mainWidget = QWidget()        
        self.gridLayout = QGridLayout()

        self.setWindowTitle("ASL Data Collection Interface")
        
        self.createWidgets()
        
        ## Set the mainWidget layout to the grid layout
        self.mainWidget.setLayout(self.gridLayout)
        ## Set the central widget
        self.setCentralWidget(self.mainWidget)          

        ## Create a status bar to display info to user
        self.status = self.statusBar()
        self.status.setSizeGripEnabled(False)
        
        self.status.showMessage("Ready",5000)
        self._defaultColor = 'rgb(240,240,240)'
        self._redColor = 'rgb(255,0,0)'
        self._greenColor = 'rgb(0,255,0)'
                
        
    def createWidgets(self):
        
        """This function calls the functions responsible for building the GUI
        
            PARAMS: self: self
            
            RETURNS: NONE"""
        
        self.createImageViews()
        self.createUserOptions()
        self.createCharOptions()
        self.createStartPreview()
        
    def createImageViews(self):
        
        """This function creates the QLabel items responsible for holding the 
           images received from the sensor.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""
           
        self.imageLabelL = QLabel("Left Image")
        self.imageLabelR = QLabel("Right Image")
        self.imageViewL = QLabel("L Image Holder")
        self.imageViewL.setMinimumSize(QSize(640,240))
        self.imageViewR = QLabel("R Image Holder")
        self.imageViewR.setMinimumSize(QSize(640,240))
        
        self.gridLayout.addWidget(self.imageLabelL,0,0)
        self.gridLayout.addWidget(self.imageLabelR,0,2)
        self.gridLayout.addWidget(self.imageViewL,1,0,1,2)
        self.gridLayout.addWidget(self.imageViewR,1,2,1,2)
        
        self.gridLayout.setRowStretch(0, 0)
        self.gridLayout.setRowStretch(1, 0)
        
    def createUserOptions(self):
        
        """This function creates the user information items, and adds those items
           to the GUI layout.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""

        self.userLabel = QLabel("User: ")
        
        self.userGroupBox = QGroupBox("User")
        
        self.userBox = QGridLayout()
        self.userBox.addWidget(self.userLabel,0,0)
        
        self.newUserBtn = QRadioButton("New User")
        self.newUserBtn.setChecked(True)
        try:
            self.connect(self.newUserBtn,SIGNAL("clicked()"),self.populateUserSpin)
        except AttributeError:
            self.newUserBtn.clicked.connect(self.populateUserSpin)
        self.existingUserBtn = QRadioButton("Existing User")
        self.existingUserBtn.setChecked(False)
        try:
            self.connect(self.existingUserBtn,SIGNAL("clicked()"),self.populateUserSpin)
        except AttributeError:
            self.existingUserBtn.clicked.connect(self.populateUserSpin)
        
        self.userBox.addWidget(self.newUserBtn,0,1)
        self.userBox.addWidget(self.existingUserBtn,0,2)
                
        self.userIDLabel = QLabel("ID Number:")
        self.userIDSpin = QSpinBox()
        self.userIDSpin.lineEdit().setReadOnly(True)

        self.userBox.addWidget(self.userIDLabel,1,0)
        self.userBox.addWidget(self.userIDSpin,1,1,1,2)                        
        
        self.userGroupBox.setLayout(self.userBox)
        
        self.gridLayout.addWidget(self.userGroupBox,2,0)
        self.gridLayout.setRowStretch(2, 1)
        self.populateUserSpin()

    def createCharOptions(self):
        
        """This function creates the character information items, and adds those items
           to the GUI layout.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""
        
        self.charLabel = QLabel("Selected Character: ")
        self.handLabel = QLabel("Handedness: ")
        
        self.charGroupBox = QGroupBox("Static Character")
        
        self.charBox = QGridLayout()
        
        self.charBox.addWidget(self.handLabel,0,0,1,1)
        self.charBox.addWidget(self.charLabel,1,0,1,1)
        
        self.charCombo = QComboBox()
        self.charCombo.addItems(self._staticChars)
        self.charBox.addWidget(self.charCombo,1,1,1,2)
        
        self.leftHandBtn = QRadioButton("Left")
        self.rightHandBtn = QRadioButton("Right")
        self.rightHandBtn.setChecked(True)
        
        self.charBox.addWidget(self.leftHandBtn,0,1,1,1)
        self.charBox.addWidget(self.rightHandBtn,0,2,1,1)
        
        self.charGroupBox.setLayout(self.charBox)
        self.gridLayout.addWidget(self.charGroupBox)
        
    def createStartPreview(self):
        
        """This function creates the buttons for data collection and camera
           preview, and adds those items to the GUI layout.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""

        self.previewCamBtn = QPushButton("Preview Camera")
        try:
            self.connect(self.previewCamBtn,SIGNAL("clicked()"),self._camPreview.startCamera)
        except AttributeError:
            self.previewCamBtn.clicked.connect(self._camPreview.startCamera)
        self.stopCamBtn = QPushButton("Stop Camera")
        try:
            self.connect(self.stopCamBtn,SIGNAL("clicked()"),self._camPreview.stopCamera)
        except AttributeError:
            self.stopCamBtn.clicked.connect(self._camPreview.stopCamera)
        self.startCollectBtn = QPushButton("Collect Data")
        try:
            self.connect(self.startCollectBtn,SIGNAL("clicked()"),self.collectData)
        except AttributeError:
            self.startCollectBtn.clicked.connect(self.collectData)
        
        self.buttonBox = QGridLayout()
        self.buttonBox.addWidget(self.previewCamBtn,0,0)
        self.buttonBox.addWidget(self.stopCamBtn,0,0)
        self.stopCamBtn.hide()
        self.buttonBox.addWidget(self.startCollectBtn,0,1)
        
        self.gridLayout.addLayout(self.buttonBox,3,0,1,2)
        
    def populateUserSpin(self):

        """This function populates the user spin dial with the options for user
           id number.  This number is determined by user status provided through
           the New User and Existing User radio buttons.  If an existing user
           is returning to finish data selection, the options available are 
           determined by searching the DATA output path for user directories.
           If the user is new, the option available is the next ID integer
           
           PARAMETERS:  self: self
           
           RETURNS:  NONE"""        
        
        self.userIDSpin.cleanText()
        existingDirs = os.listdir(self._dataPath)
        existingDirs.sort()
        
        numbers = []
        for d in existingDirs:
            if d.startswith('User_'):
                numbers.append(int(d[d.rfind("_")+1:]))
        
        if self.newUserBtn.isChecked():
            self.userIDSpin.cleanText()
            nextID = 1 if len(numbers) == 0 else numbers[-1] + 1
            self.userIDSpin.setRange(nextID,nextID)
            self.userIDSpin.setValue(nextID)
            self.userIDSpin.lineEdit().setText(str(nextID))
            self.userIDSpin.setReadOnly(True)
        else:
            self.userIDSpin.cleanText()
            self.userIDSpin.setRange(numbers[0],numbers[-1])
            self.userIDSpin.setReadOnly(False)
            self.userIDSpin.lineEdit().setReadOnly(True)
            
        
    def forceCamQuit(self):

        """This function check to see if the QThread is running, and if so stops
           the thread and re-starts it so that the preview can be started again.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""

        if self._camPrevThread.isRunning():
            self._camPrevThread.quit()
            self._camPrevThread.wait()
            self._camPrevThread.start()

    
    @pyqtSlot(list)
    def updateImage(self,images):
        
        """This function converts the numpy array image data into a QImage object
           that is then set to a QPixmap object to be placed on the GUI.
           
           PARAMETERS:  images: list
                           a list that contains the numpy arrays of the image 
                           data from the Leap Sensor.
                           
           RETURNS:  NONE"""
       
        if images[0] is not None:
            message = images[-1]
            self.previewCamBtn.hide()
            self.stopCamBtn.show()
            limg = images[0]
            rimg = images[1]
            qimgL = QImage(limg.data,limg.shape[1],limg.shape[0],QImage.Format_Indexed8)
            qimgR = QImage(rimg.data,rimg.shape[1],rimg.shape[0],QImage.Format_Indexed8)
            self._pixmapL = QPixmap.fromImage(qimgL)
            self._pixmapR = QPixmap.fromImage(qimgR)
            self.imageViewL.setPixmap(self._pixmapL)     
            self.imageViewR.setPixmap(self._pixmapR) 
            #print self.imageViewL.frameGeometry().width()
            #print self.imageViewL.frameGeometry().height()     
           
            self.updateStatusBar(message)
           
        else:
            self.stopCamBtn.hide()
            self.previewCamBtn.show()
            self.forceCamQuit()
            self.status.clearMessage()
            self.status.setStyleSheet("QStatusBar{background:%s}" % self._defaultColor)
    
    def updateStatusBar(self,message):
        
        currMessage = self.status.currentMessage()

        if currMessage != message:
            self.status.clearMessage()
            if message == "Hand in position":
                self.status.setStyleSheet("QStatusBar{background:%s}" % self._greenColor)
            else:
                self.status.setStyleSheet("QStatusBar{background:%s}" % self._redColor)
            self.status.showMessage(message)
    
    def collectData(self):
        
        """This function will start the data recording process.  It will create
           the data directories for new users.  It will collect 10 frames for 
           the user selected static character."""        
        
        handInPos = True
        print "Collecting Data Called!!"
        count = 0
        self.initOutputDirs()
        self.startCollectBtn.setText("Collecting Data...")
        self.startCollectBtn.setEnabled(False)
        QApplication.processEvents()
        while count < self._numFramesToCollect:
            while(not self._controller.is_connected):
                print "connecting..."
                pass

            frame = self._controller.frame()
            if self.iBox is None:
                self.iBox = frame.interaction_box
#            self.iBoxWidth = self.iBox.width # x-axis
#            self.iBoxHeight = self.iBox.height # y-axis
#            self.iBoxDepth = self.iBox.depth # z-axis
#            self.iBoxCenter = self.iBox.center # vector
#            msg = ""
            if ((frame.is_valid) and (not frame.images.is_empty) and
                (not frame.hands.is_empty) and (not frame.fingers.is_empty)):
                l_img, r_img = lutils.images_to_np_arrays(frame.images)
                
                if frame.hands.is_empty:
                    msg = "No Hands in Frame"
                else:
                    palmPos = frame.hands[0].palm_position
                    msg = lutils.putHandInIBox(palmPos,self.iBox)
#                    ## Assuming orientation is facing user,light down (bottom right corner),plug on left
#                    if palmPos.z > (self.iBoxCenter.z + (self.iBoxDepth / 2)):
#                        msg = "Move hand up"
#                    if palmPos.z < (self.iBoxCenter.z - (self.iBoxDepth / 2)):
#                        msg = "Move hand down"
#                    if palmPos.x > (self.iBoxCenter.x + (self.iBoxWidth / 2)):
#                        if msg == "":
#                            msg = "Move hand left"
#                        else:
#                            msg = msg + ", left"
#                    if palmPos.x < (self.iBoxCenter.x - (self.iBoxWidth / 2)):
#                        if msg == "":
#                            msg = "Move hand right"
#                        else:
#                            msg = msg + ", right"
#                    if palmPos.y > (self.iBoxCenter.y + (self.iBoxHeight / 2)):
#                        if msg == "":
#                            msg = "Move hand forward"
#                        else:
#                            msg = msg + ", forward"
#                    if palmPos.y < (self.iBoxCenter.y - (self.iBoxHeight / 2)):
#                        if msg == "":
#                            msg = "Move hand backward"
#                        else:
#                            msg = msg + ", backward"
#                    if msg == "":
#                        msg = "Hand in position"
                if msg != "Hand in position":
                    handInPos = False
                else:
                    handInPos = True
                self.updateImage([l_img,r_img,msg])
                QApplication.processEvents()
                if handInPos:
                    #serialized = lutils.serializeData(frame)
                    self.saveData(frame,l_img,r_img,count)
                    count += 1
                    print "Count",count
        self.startCollectBtn.setText("Collect Data")
        self.startCollectBtn.setEnabled(True)
        
        print "Data Collected"
        
    def saveData(self,frame,limg,rimg,count):
        
        frameDir = self.userDir + "\\Serialized"
        imgDir = self.userDir + "\\Images"
        
        character = str(self.charCombo.currentText())
        hand = "RH" if self.rightHandBtn.isChecked() else "LH"
        lname = "%s_Left_Image_%s_%s.jpg" % (hand,str(count).zfill(3),character.upper())
        rname = "%s_Right_Image_%s_%s.jpg" % (hand,str(count).zfill(3),character.upper())
        scipy.misc.toimage(limg, cmin=np.min(limg), cmax=np.max(limg)).save(imgDir+"\\"+lname)
        scipy.misc.toimage(rimg, cmin=np.min(rimg), cmax=np.max(rimg)).save(imgDir+"\\"+rname)

        
        fname = "%s_Frame_%s_%s.data" % (hand,str(count).zfill(3),character.upper())
        serialized_tuple = frame.serialize
        serialized_data = serialized_tuple[0]
        serialized_length = serialized_tuple[1]
        data_address = serialized_data.cast().__long__()
        bfr = (ctypes.c_ubyte * serialized_length).from_address(data_address)
        serialPath = frameDir + "\\" + fname
        with open(serialPath, 'wb') as data_file:
            data_file.write(bfr)
        
    def initOutputDirs(self):
        
        """This function will create the output directories for new users."""
        if self.newUserBtn.isChecked():
            userID = "User_" + str(self.userIDSpin.value()).zfill(2)
            self.userDir = self._dataPath + "\\" + userID
            try:
                os.mkdir(self.userDir)
                for d in self._dataFolders:
                    outPath = self.userDir + '\\' + d
                    os.mkdir(outPath)
            except:
                pass
        else:
            userID = "User_" + str(self.userIDSpin.value()).zfill(2)
            self.userDir = self._dataPath + "\\" + userID
        

    def closeEvent(self,event):
        
        """This function is makes sure the Thread object is properly termintated
           before shutting down the GUI.
           
           PARAMETERS:   self: self
                         event : QEvent
                            The event that triggers the function (application close)"""
        
        self._camPrevThread.terminate()
        event.accept()
        
if __name__ == "__main__":
    app=QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()
