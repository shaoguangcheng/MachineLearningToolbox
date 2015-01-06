#!/usr/bin/env python

import decisionTree as dt

def createDataset() :
	dataset = [[1, 1],
				[1, 1],
				[1, 0],
				[0, 1],
				[0, 1]]
	labels = ['yes', 'yes', 'no', 'no', 'no']
	return (dataset, labels)

if __name__ == '__main__' :
	(dataset, labels) = createDataset()
	labelName = ["no surface", "flipper"]

	tree = dt.createTree(dataset, labels, labelName)
	print(tree)

	labelName = ["no surface", "flipper"]
	label = dt.classify(tree, labelName, [0,0])
	print(label)
