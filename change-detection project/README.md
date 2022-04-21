# Computer Vision Challenge Group36

## Environmental requirements
* MATLAB Version: 	MATLAB 2020 or newer
* Required toolbox:	Computer Vision Toolbox
			Image Processing Toolbox

## Introduction to our folder
change-detection project
	 |- main.m
	 |- gui.mlapp
	 |- Readme.txt
	 |- datesets
	 |- change_analysis
	 |- line_detection
     |- image_registration


## Overview
This program is developed to show change in satellite images. The function is realized through the MATLAB programs in the folder 'G36'.

* main.m: is the main function
* gui.mlapp: is the GUI Application
* Readme.txt
* datasets: consists different subfolders of image data
* change_analysis: folder of functions to analyze the change, including: classification, differentiation...
* line_detection: folder of functions to detect number of lines in the image
* image_registration: folder of functions to accomplish the image registration and gif animation


## Getting Started
* For a quick start, run 'main.m'.
* To run the application, run 'gui.mlapp' in MATLAB.

## Launch the .m program
Follow the instructions in the command window and the results can be shown after execution.

## Launch the APP:
Step 1
Add all the folders into the path of your MATLAB.

Step 2
Run the App and select one subfolder in the folder 'Datasets', e.g. Dubai...

Step 3
First select one option as reference image in the drop-down menu.

Step 4
Second select one option as target image in the drop-down menu.

# Notice: To see the complete function of the Application, you should obey the order:

          Illumination Normalization-Registration of Selected Images-Differentiation-Highlight/Evaluate

* Step 5
Trigger the button 'Illumination Normalization'. The illumination of two images will be unified.

* Step 6
Trigger the button 'Registration of Selected Images'. The target image is then be registered and is aligned to the reference image

* Step 7
Trigger the button 'Differentiation'. The change between two images is detected. In the output figure, white part is denotes the change.

The Differentiation Threshold is adjustable and you can apply different value of this threshold and get different Differentiation outputs.
After you change the Differentiation Threshold, you should trigger the button 'Differentiation' again to see new output.

# Notice:
After Step 7, the remaining Steps here (Step8, Step9，Step 10 and Step 11) can be executed in any orders you like. Here is only one option presented.

** Step 8
Trigger the button 'highlight', you can see the change part are highlighted in the red color.

** Step 9
In the section Classification', trigger the button 'Evaluate' and you can see wether the changed area belongs to nature or urban change.

** Step 10
In the section 'Quantitative Indicators', trigger the button 'Evaluate' and the values of 'Changing Time', 'Changing ratio' and 'Changing' speed are shown. 

** Step 11
Trigger the button "Reset", You can clear the data remaining in the application.

***Step 12
Have fun with the Application :)

# Notice:
The function of 'Registration of All Images' and 'Show Time Lapse' can be realized after the subfolder is selected.

When you want to see the 'Animation' e.g.'Registration of All Images', it may takes relatively longer time, because it takes time to execute registration of all images.
So please wait a little bit if you trigger the button 'Registration of All Images'.

## Already Available Datasets:
* Dubai
* Kuwait
* Frauenkirche
* Wiesn
* Columbia Glacier
* Aral Sea
