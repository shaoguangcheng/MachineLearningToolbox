#!/usr/bin/env python

import os
from numpy import *

import knn

# convert image to vector
def  img2vector(filename):
 	rows = 32
 	cols = 32
 	imgVector = zeros((1, rows * cols)) 
 	fileIn = open(filename)
 	for row in xrange(rows):
 		lineStr = fileIn.readline()
 		for col in xrange(cols):
 			imgVector[0, row * 32 + col] = int(lineStr[col])

 	return imgVector

# load dataSet
def loadDataSet(datapath):
	print "---Getting training set..."
	dataSetDir = datapath
	trainingFileList = os.listdir(dataSetDir + 'trainingDigits') # load the training set
	numSamples = len(trainingFileList)

	train_x = zeros((numSamples, 1024))
	train_y = []
	for i in xrange(numSamples):
		filename = trainingFileList[i]

		train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename) 

		# get label from file name such as "1_18.txt"
		label = int(filename.split('_')[0])
		train_y.append(label)

	print "---Getting testing set..."
	testingFileList = os.listdir(dataSetDir + 'testDigits') # load the testing set
	numSamples = len(testingFileList)
	test_x = zeros((numSamples, 1024))
	test_y = []
	for i in xrange(numSamples):
		filename = testingFileList[i]

		# get train_x
		test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename) 

		# get label from file name such as "1_18.txt"
		label = int(filename.split('_')[0]) # return 1
		test_y.append(label)

	return train_x, train_y, test_x, test_y

def testHandWritingClass(datapath):
	print "step 1: load data..."
	train_x, train_y, test_x, test_y = loadDataSet(datapath)

	print "step 2: training..."
	pass

	print "step 3: testing..."
	numTestSamples = test_x.shape[0]
	matchCount = 0
	for i in xrange(numTestSamples):
		predict = knn.kNNClassify(test_x[i], train_x, train_y, 3)
		if predict == test_y[i]:
			matchCount += 1
	accuracy = float(matchCount) / numTestSamples

	print "step 4: show the result..."
	print 'The classify accuracy is: %.2f%%' % (accuracy * 100)

if __name__ == "__main__" :
	datapath = "../data/"
	testHandWritingClass(datapath)
