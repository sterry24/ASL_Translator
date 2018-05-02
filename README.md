# ASL Translator
A project to 
  * collect ASL (American Sign Language) static gestures through the use of the Leap Motion Sensor.<br>
  * translate ASL gestures into text with the use of a machine learned model from data collection

#### Features:
Currently reads files of the following formats:<br>
  * Excel<br>
  * comma delimited or white-space delimited<br>

#### Includes:
  * One RandomForest model created from data collected from 6 users<br>
  * leap_utils.py:<br>
	  * Some utilities for working with leap motion data<br>

Built-in Graph Generator for plotting data <br>
Merge tool for merging multiple dataframes
Group-by tool for selecting subsets of data


#### Requirements:
Python 2.7<br>
PyQt4 or PyQt5<br>
Leap Motion Sensor<br>
Leap Motion Sensor SDK<br>




#### Screenshots
ASL_dataCollectGUI<br>
  * Modify lines 9 and 10 for path to SDK (LeapDeveloperKit_3.2.0+45899_win\LeapDeveloperKit_3.2.0+45899_win\LeapSDK\lib)<br>
  * Modify line 157 to update output paths for data collection<br>
![](./screenshots/DataCollectGUI.png)<br>

ASL_TranslatorGUI<br>
  * Modify lines 9 and 10 for path to SDK (LeapDeveloperKit_3.2.0+45899_win\LeapDeveloperKit_3.2.0+45899_win\LeapSDK\lib)<br>
  * Modify line 54 to update model path<br>
![](./screenshots/TranslateGUI.png)<br>