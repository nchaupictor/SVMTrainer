# ROI Cropping Python Script
# Auto detects wells and partitions into a single stitched image of RGB channels,
# individual RGB wells and Greyscale wells. Raw intensity data of each slide gets
# compressed into a (x*y size of well crop) * (16 wells) array and saved into a csv 
# A 16 x 1 array of labels is to be generated which indicates which wells are wet (1) or dry (0)
# Author: N Chau
# Date: 26/05/2016 Updated: 27/02/2017

# Import libraries
# ------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

#from urllib import urlretrieve
#import cPickle as cPickle
import os 
import gzip

import numpy as np 
#import theano
import time
from PIL import Image

#import lasagne
#from lasagne import layers
#from lasagne.updates import nesterov_momentum

#from nolearn.lasagne import NeuralNet
#from nolearn.lasagne import visualise

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import cv2 

from operator import itemgetter
# ------------------------------------------------------------------------
# Read in file from sql / file
filenameA = 'ENA 20 D_'
filename = filenameA + '.bmp'
img = cv2.imread(filename)
imgGray = cv2.imread(filename,0)
imgWell = img.copy()
cv2.imshow('image',img)
cv2.waitKey(0)

# Pre-process image 
imgGrayB = cv2.cvtColor(imgGray,cv2.COLOR_GRAY2BGR)

#cv2.imshow('gray',imgGray)
#cv2.waitKey(0)


# ------------------------------------------------------------------------
# Run circle detection on slide
circles = cv2.HoughCircles(imgGray,cv2.HOUGH_GRADIENT, 0.6, 200,param1 = 20, param2 = 20, minRadius = 93, maxRadius = 100)
circlesCoord = circles[0,:].astype(int)

print(circlesCoord[0:16,0:2]) #row then col
print(circlesCoord.shape)


# Sort circle coordinates
circlesCoordS = np.ndarray(shape=(16,2),dtype = int)
circlesCoordSB = np.ndarray(shape=(16,2),dtype = int)
a = sorted(circlesCoord, key = itemgetter(1))
#print(a[15][1])

print('Length: ')
print(len(a))

classCol = np.ndarray(shape=(len(a),1),dtype = int)
# 1 = Wet
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Set this string to either 0 or 1 to save into labels.csv at the end 
classString = 0

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
for k in range(0,len(a)):
	classCol[k] = classString
print(classCol)

for i in range(0,len(a)):
	circlesCoordS[i,0:2] = a[i][0:2]
print(circlesCoordS)

for i in range(0,len(a),2):
	a = sorted(circlesCoordS[i:(i+2)], key = itemgetter(0))
	#print(i)
	#print(a[0][0:2])
	#print(a[1][0:2])
	#circlesCoordS[i,0:2] = a[0][0:2]
	#print(circlesCoordS[i,0:2])
	#circlesCoordS[i+1,0:2] = a[1][0:2]
	#print(circlesCoordS[i+1,0:2])
	circlesCoordSB[i,:] = a[0][0:2]
	#print(circlesCoordSB[i])
	circlesCoordSB[i+1,:] = a[1][0:2]
	
print(circlesCoordSB)

if circles is not None:
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		cv2.circle(imgGrayB,(i[0],i[1]),i[2],(0,255,0),2)

	cv2.namedWindow('imgWell', cv2.WINDOW_NORMAL)
	cv2.imshow("imgWell",imgGrayB)
	#cv2.waitKey(0)
	cv2.destroyAllWindows()


# ------------------------------------------------------------------------
# Crop slide into 16 wells of 110 x 110 x 3 (RGB)
step = 95 #55
#ROI = np.ndarray(shape=(110,110,3),dtype = int)
#ROIvec = np.ndarray(shape = (len(classCol),110,330), dtype = int)
#ROIflat = np.ndarray(shape = (len(classCol),110*330), dtype = int)
ROI = np.ndarray(shape=(step*2,step*2,3),dtype = int)
ROIg = np.ndarray(shape = (step*2,step*2),dtype = int)
ROIvec = np.ndarray(shape = (len(classCol),step*2,step*2*3), dtype = int)
ROIflat = np.ndarray(shape = (len(classCol),step*2*step*2*3), dtype = int)
ROIvecG = np.ndarray(shape = (len(classCol),step*2*step*2,1), dtype = int)
ROIflatG = np.ndarray(shape = (len(classCol),step*2*step*2*1), dtype = int)

print(circlesCoordSB[0][1])

#Create folder to store ROI images and csv
pathA = r'C:\Users\Pictor17\Python'
pathB = pathA +'\\' + filenameA
print('Directory: ')
print(pathB)
if not os.path.exists(pathB):
	os.makedirs(pathB)
#Navigate to folder 
os.chdir(pathB)

for i in range(0,len(classCol)):
	print('ROI Length: ')

	print(len(classCol))
	ROI = img[circlesCoordSB[i][1]-step:circlesCoordSB[i][1]+step,circlesCoordSB[i][0]-step:circlesCoordSB[i][0]+step]
	cv2.namedWindow('ROIRGB',cv2.WINDOW_NORMAL)
	cv2.imshow('ROIRGB',ROI)
	#cv2.waitKey(0)
	#wellRGB = Image.fromarray(ROI).convert('RGB')
	#strtemp = 'WellRGB' + str(i+1) + '.bmp'
	#wellRGB.save(strtemp)
	ROIg = imgGray[circlesCoordSB[i][1]-step:circlesCoordSB[i][1]+step,circlesCoordSB[i][0]-step:circlesCoordSB[i][0]+step]
	cv2.imwrite('WellG' + str(i+1) + '.bmp',ROIg)
	cv2.imwrite('WellRGB' + str(i+1) + '.bmp',ROI)

	#Resize ROIg to 

	print(ROI.size)
	print(np.size(ROI,0))
	print(np.size(ROI,2))

	print(np.size(ROIg,0))
	print(np.size(ROIg,1))
	temp = np.concatenate((ROI[:,:,0],ROI[:,:,1],ROI[:,:,2]),axis = 1)
	temp2 = np.reshape(ROIg,[step*2*step*2,1])
	# Store / save wells
	well = Image.fromarray(temp)
	strtemp = 'Well' + str(i+1) + '.bmp'
	print(strtemp)
	
	well.save(strtemp)
	ROIvec[i] = temp
	print(np.size(ROIvec,0))
	print(np.size(ROIvec,1))
	print(np.size(ROIvec,2))

	ROIvecG[i] = temp2


	# Reshape 2D ROI plane (110 x 330) into single vectors (1 x 110 x 330)

	ROIflat[i] = ROIvec[i].flatten()
	ROIflatT = np.transpose(ROIflat)
	#ROIflatT = np.divide(ROIflatT,255)
	ROIflatG[i] = ROIvecG[i].flatten()
	ROIflatTG = np.transpose(ROIflatG)


	print(np.size(ROIflat,0))
	print(np.size(ROIflat,1))
	print(ROIflat[i])

	cv2.namedWindow('ROI',cv2.WINDOW_NORMAL)
	cv2.imshow('ROI',temp) 
	#cv2.waitKey(0)
	cv2.destroyAllWindows()
#Save ROIflat into csv	
np.savetxt(filenameA + 'dataG.csv', ROIflatTG, delimiter = ",")

np.savetxt(filenameA + 'data.csv',ROIflatT,delimiter=",")
print(ROIflatT)
np.savetxt(filenameA + 'labels.csv',classCol,delimiter=",")

# Normalise/ Standardise intensities to mu and sigma 



# ------------------------------------------------------------------------
# Add classification labels to each well(s)



# Split dataset into 90% training 10% testing 





# Set up convolutional neural network (CNN) 



# Train CNN




# Predict with test set




# Create model visuals 





