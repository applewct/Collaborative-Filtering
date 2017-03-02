import time
from numpy import genfromtxt
import pandas as pd
import numpy as np
import scipy.sparse as ss
import math
from sklearn.model_selection import train_test_split

def topN (a,N):
    return np.argsort(a)[::-1][:N]

def read2array(inputFile):
    f = open(inputFile, 'r')

    row = []
    col = []
    data = []
    test = []

    for line in f:
        user, movie, rating = line.split(' ')
        user  = int(user)
        movie = int(movie)
        rating = int(rating)
        movie -= 1
        if (rating == 0):
            test.append((user,movie))
            continue
        row.append(user)
        col.append(movie)
        data.append(rating)
    return row, col, data, test

def save2sparse(inputFile):
    row = []
    col = []
    data = []
    test = []

    tmprow, tmpcol, tmpdata, tmpTest = read2array(inputFile)
    row.extend(tmprow)
    col.extend(tmpcol)
    data.extend(tmpdata)
    test.extend(tmpTest)
    trainMat = ss.csr_matrix((data, (row, col)))
    return trainMat, test

def cosSim(userVec, row):
    uRows,uCols = userVec.nonzero()
    nu = 0
    denoL = 0
    denoR = 0
    for i in uCols:
        nu += userVec[uRows[0], i] * row[i]
        denoL += math.pow(userVec[uRows[0], i], 2)
        denoR += math.pow(row[i], 2)
    if (denoR < 0.00001):
        return 0
    return nu/((math.sqrt(denoL))*(math.sqrt(denoR)))

def cosPredict(userVec, fullTrain, movie, num):
    dist = []
    for row in fullTrain:
        if(int(row[movie]) == 0):
            dist.append(0)
        else:
            dist.append(cosSim(userVec, row))
    kDist = topN(dist, num)
    nu = 0.0
    de = 0.0
    for i in range(len(kDist)):
        # print dist[kDist[i]]
        nu += dist[kDist[i]] * fullTrain[kDist[i]][movie]
        de += dist[kDist[i]]
    if (nu < 0.00001):
        return 3
    return nu/de

def pearSim(userVec, row):
    uRows,uCols = userVec.nonzero()
    nu = 0.0
    denoL = 0.0
    denoR = 0.0
    uAvg = 0.0
    trainAvg = 0.0
    for i in uCols:
        # calculate u avg
        uAvg += userVec[uRows[0], i]
        # calculate trainAvg
        trainAvg += row[i]
    uAvg /= len(uCols)
    trainAvg /= len(uCols)
    for i in uCols:
        nu += (userVec[uRows[0], i]-uAvg) * (row[i]-trainAvg)
        denoL += math.pow((userVec[uRows[0], i]-uAvg), 2)
        denoR += math.pow((row[i]-trainAvg), 2)
    if denoR < 0.00001 or denoL < 0.00001:
        return 0.0
    return nu/((math.sqrt(denoL))*(math.sqrt(denoR)))

def pearPredict(userVec, fullTrain, movie, num):
    uAvg = 0.0
    uRows,uCols = userVec.nonzero()
    for i in uCols:
        # calculate u avg
        uAvg += userVec[uRows[0], i]
    uAvg /= len(uCols)

    dist = []
    for row in fullTrain:
        if(int(row[movie]) == 0):
            dist.append(0)
        else:
            dist.append(pearSim(userVec, row))
    unSignedDist = []
    for i in dist:
        unSignedDist.append(abs(i))
    kDist = topN(unSignedDist, num)
    nu = 0.0
    de = 0.0
    for i in range(len(kDist)):
        # calculate trainAvg
        trainAvg = np.mean(fullTrain[kDist[i]])
        nu += dist[kDist[i]] * (fullTrain[kDist[i]][movie] - trainAvg)
        de += unSignedDist[kDist[i]]
    if (nu < 0.00001):
        return uAvg
    return nu/de + uAvg

def runPredict(inputFile, outputFile, predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    allTest, testCo = save2sparse(inputFile)
    start = time.time()
    out = open(outputFile, 'w')
    for co in testCo:
        print co[0]
        rating = predictFunc(allTest[co[0]], fullTrain, co[1], 10)
        rating = int(round(rating))
        if (rating > 5):
            rating = 5
        if (rating < 1):
            rating = 1
        out.write(str(co[0]) + " " + str(co[1] + 1)+ " " + str(rating) + "\n")
    end = time.time()
    print "time used to run: " , end - start

def runTesting(predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    # df = pd.DataFrame(fullTrain)
    train, test = train_test_split(fullTrain, test_size = 0.2)
    print train.shape
    # create test
    test5 = []
    ans = []
    testIdx = []
    for oneTest in test:
        tmpTest5 = []
        tmpAns = []
        tmpTestIdx = []
        testNum = 0
        for i in range(1000):
            # create test with only 5 rating
            if testNum < 5:
                tmpTest5.append(oneTest[i])
            else:
                tmpTest5.append(0.0)
            if oneTest[i] > 0.0001:
                testNum += 1
            if oneTest[i] > 0.0001 and testNum >= 5:
                # log testIdx
                tmpTestIdx.append(i)
                # log ans
                tmpAns.append(oneTest[i])
        test5.append(tmpTest5)
        ans.append(tmpAns)
        testIdx.append(tmpTestIdx)



    # allTest, testCo = save2sparse(inputFile)
    # start = time.time()
    # out = open(outputFile, 'w')
    # for co in testCo:
    #     print co[0]
    #     rating = predictFunc(allTest[co[0]], fullTrain, co[1], 10)
    #     rating = int(round(rating))
    #     if (rating > 5):
    #         rating = 5
    #     if (rating < 1):
    #         rating = 1
    #     out.write(str(co[0]) + " " + str(co[1] + 1)+ " " + str(rating) + "\n")
    # end = time.time()
    # print "time used to run: " , end - start

intputList = ["test5.txt", "test10.txt", "test20.txt"]
outputList = ["result5.txt", "result10.txt", "result20.txt"]
i = 0
runTesting(intputList[i], outputList[i], pearPredict)
