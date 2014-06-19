from numpy import *

## helper functions for the SMO
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


## simplified SMO algorithm
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #fXi is prediction of the class
            fXi = float(multiply(alphas,labelMat).T* \
                        (dataMatrix*dataMatrix[i,:].T)) + b
            #Ei is error
            Ei = fXi - float(labelMat[i])
            #if error is too large then alpha should be optimized
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
               ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #1. randomly select j
                j = selectJrand(i,m)
                #2. recompute fXj and Ej
                fXj = float(multiply(alphas,labelMat).T*\
                            (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #3. compute L and H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                #4. Eta is the optimal amount to change alpha[j]
                #   this step is simplification of real SMO
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
                      dataMatrix[i,:] * dataMatrix[i,:].T - \
                      dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                #5. calculate alpha and clip it
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #6. check if alpha changed small amount
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"; continue
                #7. alpha[i] changed in the opposite direction
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                #8. finally set the constants b with alphas
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[i,:].T - \
                     labelMat[j]*(alphas[j]-alphaJold) * \
                     dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[j,:].T - \
                     labelMat[j]*(alphas[j]-alphaJold) * \
                     dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i: %d, pairs changed %d" % \
                      (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number : %d" % iter
    return b,alphas
