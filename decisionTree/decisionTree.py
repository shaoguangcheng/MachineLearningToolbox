#!/usr/bin/env python

import numpy as np

def calcEntropy(labels) :
	'''calculate the entropy according to data distribution\

	   H(p) = -sum(Pi*log(Pi))'''
	num = len(labels)
	labelCount = {}
	for label in labels :
		if label not in labelCount.keys() :
			labelCount[label] = 0;
		labelCount[label] += 1

		#labelCount[label] = labelCount.get(label, 0) + 1

	entropy = 0.0
	for val in labelCount.values() :
		prob = val*1.0/num
		entropy += prob*np.log2(prob)

	return -1.0*entropy

def splitDataset(dataset, label, f, v) :
	'''split dataset using feature f with its value v'''
	resultDataset = []
	resultLabel = []
	for index, example in enumerate(dataset) :
		if example[f] == v :
			feature = example[:f]
			feature.extend(example[f+1:])
			resultDataset.append(feature)
			resultLabel.append(label[index])
	return (resultDataset, resultLabel)

def chooseBestFeatureToSplit(dataset, label) :
	num = len(label)
	nfeature = len(dataset[0])
	baseEntropy	= calcEntropy(label)
	bestInfoGain = 0.0
	bestFeature = -1
	for index in range(nfeature) :
		featureList = [example[index] for example in dataset]
		uniqueVal = set(featureList)
		newEntropy = 0.0
		for v in uniqueVal :
			(subset, sublabel) = splitDataset(dataset, label, index, v)
			prob = len(sublabel)*1.0/num
			newEntropy += prob*calcEntropy(sublabel)

		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain :
			bestInfoGain = infoGain
			bestFeature = index

	return bestFeature

def createTree(dataset, label, labelName) :
	if label.count(label[0]) == len(label) :
		return label[0]

	bestFeature = chooseBestFeatureToSplit(dataset, label)
	bestFeatureName = labelName[bestFeature]
	tree = {bestFeatureName : {}}

	tmpLabelName = labelName
	del(labelName[bestFeature])

	featureList = [example[bestFeature] for example in dataset]
	uniqueFeatureVal = set(featureList)
	sublabelName = labelName[:]
	for val in uniqueFeatureVal :
		(subdataset, sublabel) = splitDataset(dataset, label, bestFeature, val)
		tree[bestFeatureName][val] = createTree(subdataset, sublabel, sublabelName)
	return tree 

def classify(tree, labelName, testVec) :
	rootName = tree.keys()[0]
	childTree = tree[rootName]
	index = labelName.index(rootName)
	for key in childTree.keys() :
		if testVec[index] == key :
			if type(childTree[key]).__name__ == 'dict' :
				label = classify(childTree[key], labelName, testVec)
			else :
				label = childTree[key]

	return label






