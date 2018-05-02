# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 15:02:53 2017

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

import numpy as np
import pandas as pd
import scipy.misc
from sklearn.externals import joblib
from collections import OrderedDict

# import the Leap API
import Leap
# import leap utilities created for this project
# Note that the code was created from Leap examples.
import LEAPUTILS.leap_utilities as lutils
import PROCESSING.wrangle_leap_data as wrangle
from ngrams import ngrams


class FrameGrabber(QObject):
    
    """This class is the 'Worker' object for thread creating for generating
       the camera preview."""
    
    signalStatus = pyqtSignal(dict)
    
    def __init__(self,parent=None):
        super(self.__class__,self).__init__(parent)
        
        self._modelFile = r"S:\Models\RF\RandomForest_Distance_ALL_clf.pkl"
        self._model = joblib.load(self._modelFile)
        self._prevFrameID = None
        self._currFrameID = None
        self.iBox = None
        self.controller = Leap.Controller()
        self.controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
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
        self._timer.start(100)
    
    def runCamLoop(self):

        """This function grabs a frame from the Leap Motion Sensor.  The images
           are pulled from the frame, converted to numpy arrays, and then sent
           back to the main event loop via a PyQt signal emission.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""
        
        frameData = {}
        frameData['image'] = None
        prediction = {'Predict':None}
        frame = self.controller.frame()
        if self.iBox is None:
            self.iBox = frame.interaction_box
        if self._prevFrameID is None:
            self._prevFrameID = frame.id
        self._currFrameID = frame.id
        #print self._currFrameID,self._prevFrameID
#        self.iBoxWidth = self.iBox.width # x-axis
#        self.iBoxHeight = self.iBox.height # y-axis
#        self.iBoxDepth = self.iBox.depth # z-axis
#        self.iBoxCenter = self.iBox.center # vector
#        msg = ""
        if not frame.images.is_empty:
            images = frame.images
            if frame.hands.is_empty:
                msg = "No Hands in Frame"
                prediction = {'Predict':None}
            else:
                palmPos = frame.hands[0].palm_position
                msg = lutils.putHandInIBox(palmPos,self.iBox)
                if msg == "Hand in position":
                    if not lutils.handMoving(frame.hands[0]):
                        if self._prevFrameID != self._currFrameID:
                            pHand = self.controller.frame(self._currFrameID - self._prevFrameID).hands[0]
                            cHand = frame.hands[0]
                            if lutils.handChanged(pHand,cHand):
                                #print "Prediction on %d, %d" % (self._currFrameID,self._prevFrameID)
                                prediction = self.predictSign(frame.hands[0])
                                self._prevFrameID = self._currFrameID
                            else:
                                prediction = {'Predict':None}
                                #self._prevFrameID = self._currFrameID
                    else:
                        prediction = {'Predict':None}
                        #self._prevFrameID = self._currFrameID
                else:
                    prediction = {'Predict':None}
                    self._prevFrameID = self._currFrameID
            if images[0].is_valid:
                try:
                    img = lutils.image_to_np_array(images[0])
                except:
                    pass
            elif images[1].is_valid:
                try:
                    img = lutils.image_to_np_array(images[1])
                except:
                    pass
            else:
                img = None
    
            frameData['image'] = img
            frameData['Prediction'] = prediction
            frameData['Message'] = msg
                     
            self.signalStatus.emit(frameData)
                     
        else:
            self.signalStatus.emit({})
            
        #self._prevFrameID = self._currFrameID
            
    
            
    def predictSign(self,hand):
        
        """This function takes the hand instance found in the data and runs it 
           through the specified model for prediction.
           
           PARAMETERS:  hand: Leap.Hand
                           the hand to process through the model
                           
           RETURNS:     prediction : dictionary
                           a dictionary containing the predicted sign and its
                           probability"""
        
        data = OrderedDict()
        orient = "Right" if hand.is_right else "Left"
        #data = lutils.getDataTransformBasis(hand,data,orient)
        data = wrangle.getDistanceData(hand,data,orient)
        #data = wrangle.getDistanceDirData(hand,data,orient)
        #data = wrangle.getDistanceDeg(hand,data,orient)
        #data = wrangle.getDataPalmOrigin(hand,data,orient)
        #data = wrangle.getReducedData(hand,data,orient)
        
        df = pd.DataFrame(data=data,index=[0])
        

        pred = self._model.predict(df)
        prob = self._model.predict_proba(df)
#        prob = prob[0][pred[0]]
        
        return({"Predict":pred[0]})#,"Probability":prob})
        
    @pyqtSlot()
    def stopCamera(self):

        """This function stops the QTimer object that is calling the runCamLoop
           function.  It then emits an empty signal that is used by the main event
           loop so that it knows the status of the thread.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""
        
        #sys.stdout.write("Stop Camera Called")
        self._timer.stop()
        self.signalStatus.emit({})

class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow,self).__init__(parent)

        self._camView = FrameGrabber()
        self._camViewThread = QThread()
        
        self._camView.moveToThread(self._camViewThread)
        self._camViewThread.start()
        self._camView.signalStatus.connect(self.updateImage)
        
        self.lastWord = ''
        self.sentence = ''
        self.inputChar = ''
        
        self.mainWidget = QWidget()        
        self.gridLayout = QGridLayout()

        self.setWindowTitle("ASL Translator")
        
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
        
        self.createImageView()
        self.createStartTranslate()
        
    def createImageView(self):
        
        """This function creates the QLabel items responsible for holding the 
           images received from the sensor.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""
           
        self.imageLabel = QLabel("Input Image")
        self.imageView = QLabel("Image Holder")
        self.imageView.setMinimumSize(QSize(640,240))

        
        self.gridLayout.addWidget(self.imageLabel,0,0)
        self.gridLayout.addWidget(self.imageView,1,0,1,2)
        
        self.gridLayout.setRowStretch(0, 0)
        self.gridLayout.setRowStretch(1, 0)
        
    def createStartTranslate(self):
        
        """This function creates the buttons for data collection and camera
           preview, and adds those items to the GUI layout.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""

        self.startCamBtn = QPushButton("Start Camera")
        try:
            self.connect(self.startCamBtn,SIGNAL("clicked()"),self._camView.startCamera)
        except AttributeError:
            self.startCamBtn.clicked.connect(self._camView.startCamera)
        self.stopCamBtn = QPushButton("Stop Camera")
        try:
            self.connect(self.stopCamBtn,SIGNAL("clicked()"),self._camView.stopCamera)
        except AttributeError:
            self.stopCamBtn.clicked.connect(self._camView.stopCamera)
        self.inputLabel = QLabel("Input :")
        self.translateLabel = QLabel("Translation:")
        self.inputEdit = QLineEdit()
        self.inputEdit.setReadOnly(True)
        self.translateEdit = QTextEdit()
        self.translateEdit.setReadOnly(True)

        
        self.buttonBox = QGridLayout()
        self.buttonBox.addWidget(self.startCamBtn,0,0,1,2)
        self.buttonBox.addWidget(self.stopCamBtn,0,0,1,2)
        self.stopCamBtn.hide()
        self.buttonBox.addWidget(self.inputLabel,1,0,1,1)
        self.buttonBox.addWidget(self.inputEdit,1,1)
        self.buttonBox.addWidget(self.translateLabel,2,0,1,1)
        self.buttonBox.addWidget(self.translateEdit,2,1)
        
        self.gridLayout.addLayout(self.buttonBox,3,0,1,2)
        
        
    def forceCamQuit(self):

        """This function check to see if the QThread is running, and if so stops
           the thread and re-starts it so that the preview can be started again.
           
           PARAMETERS:  self: self
           
           RETURNS: NONE"""

        if self._camViewThread.isRunning():
            self._camViewThread.quit()
            self._camViewThread.wait()
            self._camViewThread.start()

    
    @pyqtSlot(list)
    def updateImage(self,data):
        
        """This function converts the numpy array image data into a QImage object
           that is then set to a QPixmap object to be placed on the GUI.
           
           PARAMETERS:  data: dict
                           a dict that contains the numpy array of one of the images 
                           data from the Leap Sensor, and any translation that 
                           has occured.
                           
           RETURNS:  NONE"""

        if data != {}:
            if data["image"] is not None:
                self.startCamBtn.hide()
                self.stopCamBtn.show()
                img = data["image"]
                qimg = QImage(img.data,img.shape[1],img.shape[0],QImage.Format_Indexed8)
                self._pixmap = QPixmap.fromImage(qimg)
                self.imageView.setPixmap(self._pixmap)     
                self.updateStatusBar(data['Message'])
                self.updateTextBox(data['Prediction'])
                QApplication.processEvents()
                #print self.imageViewL.frameGeometry().width()
                #print self.imageViewL.frameGeometry().height()       
           
        else:
            self.stopCamBtn.hide()
            self.startCamBtn.show()
            self.forceCamQuit()
            self.status.clearMessage()
            self.status.setStyleSheet("QStatusBar{background:%s}" % self._defaultColor)
            #self.translateEdit.clear()
            
    
    def updateTextBox(self,predict):
        
        """This function updates the QTextBox with the current predicted sign.
        
           PARAMETERS: data: dict
                         a dictionary containing the predicted value
                         
           RETURNS:   NONE"""
        
        #print predict
        currText = self.translateEdit.toPlainText()
        
        if predict['Predict'] is not None:
            newText = currText + str(predict['Predict'])
            self.translateEdit.setText(newText)
    
    def updateTextBoxnGram(self,predict):
        
        """This function updates the QTextBox with the current predicted sign.
        
           PARAMETERS: data: dict
                         a dictionary containing the predicted value
                         
           RETURNS:   NONE"""
        
        #print predict
        currText = self.inputEdit.text()
        #print "CURR:",currText
        #self.inputChar += str(predict['Predict'])
        #print "INPUT:",self.inputChar
        
        if predict['Predict'] is not None:
            self.inputChar += str(predict['Predict']).lower()
            print "CURR:",currText
            print "INPUT:",self.inputChar
            newText = currText + self.inputChar
            if len(self.inputChar) == 1:
                self.inputEdit.setText(newText)
            else:
                self.inputEdit.setText(newText[:-2] + newText[-1])
            
            #last = ''
            #curr = ''
            #sent = ''

#        for char in source:
            #curr += char
            if ngrams.correct(newText) != newText:
                try:
                    if ngrams.correct(self.lastWord+newText) != self.lastWord + newText:
                        self.sentence += self.lastWord
                        self.lastWord = ngrams.correct(newText[:-1])
                        self.inputChar = self.inputChar
                except:
                    seg = ngrams.segment(self.lastWord+self.inputChar)
                    self.sentence += seg[:-2][0]
                    self.lastWord = seg[-2:-1][0]
                    self.inputChar = seg[-1]
            try:
                print ngrams.segment(self.sentence+self.lastWord+self.inputChar)   
                self.translateEdit.setText(ngrams.segment(self.sentence+self.lastWord+self.inputChar))
            except:
                pass
        
    def updateStatusBar(self,message):
        
        """This function updates the status bar with instructions to the user on
           where to place hands in the imagery.
           
           PARAMETERS: message: string
                           the message to be displayed
                           
           RETURNS: NONE"""
        
        currMessage = self.status.currentMessage()

        if currMessage != message:
            self.status.clearMessage()
            if message == "Hand in position":
                self.status.setStyleSheet("QStatusBar{background:%s}" % self._greenColor)
            else:
                self.status.setStyleSheet("QStatusBar{background:%s}" % self._redColor)
            self.status.showMessage(message)
        
if __name__ == "__main__":
    app=QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()