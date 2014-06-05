from numpy import *
import operator


## create dataset and labels
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


## k-Nearest Neighbors algorithm
def classify0(inX, dataSet, labels, k):
    #1, calculate euclidian distance
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    #2, sort distances instances
    sortedDistIndicies = distances.argsort()
    
    #3, get first k instances' class
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # sort the class map in desc order
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


## text record to numpy
def file2matrix(filename):
    dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    # init a zero matrix (numberOfLines rows,3 cols) to return
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    # parse line to list
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
