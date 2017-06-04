import time
from numpy import genfromtxt
import numpy as np
import scipy.sparse as ss
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

def cosSim(userVec, row, iufArray):
    uRows,uCols = userVec.nonzero()
    nu = 0.0
    denoL = 0.0
    denoR = 0.0

    for i in uCols:
        nu += userVec[uRows[0], i] * row[i] * iufArray[i]
        denoL += math.pow(userVec[uRows[0], i], 2)
        denoR += math.pow(row[i] * iufArray[i], 2)
    if (denoR < 0.00001):
        return 0.0
    return nu/((math.sqrt(denoL))*(math.sqrt(denoR)))

def calIUF(trainMat):
    iufArray = []
    trainMat = ss.csc_matrix(trainMat)
    for i in range(trainMat.shape[1]):
        oneIUF = trainMat.getcol(i).count_nonzero()
        if (oneIUF == 0):
            iufArray.append(0)
            continue
        oneIUF = math.log10(200.0/oneIUF)
        iufArray.append(oneIUF)
    return iufArray

def cosPredict(userVec, userIdx, fullTrain, movie, num, iufArray, amp, threshold):
    dist = []
    for row in fullTrain:
        if(int(row[movie]) == 0):
            dist.append(0)
        else:
            dist.append(cosSim(userVec, row, iufArray))
    kDist = topN(dist, num)
    nu = 0.0
    de = 0.0
    for i in range(len(kDist)):
        if(dist[kDist[i]] < threshold):
            continue
        nu += math.pow(dist[kDist[i]], amp) * fullTrain[kDist[i]][movie]
        de += math.pow(dist[kDist[i]], amp)
    if (nu < 0.00001):
        return 3.0
    return nu/de

def pearSim(userVec, row, iufArray):
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
        trainAvg += row[i] * iufArray[i]
    uAvg /= len(uCols)
    trainAvg /= len(uCols)
    for i in uCols:
        nu += (userVec[uRows[0], i]-uAvg) * (row[i] * iufArray[i]-trainAvg)
        denoL += math.pow((userVec[uRows[0], i]-uAvg), 2)
        denoR += math.pow((row[i] * iufArray[i]-trainAvg), 2)
    if denoR < 0.00001 or denoL < 0.00001:
        return 0.0
    return nu/((math.sqrt(denoL))*(math.sqrt(denoR)))

def pearPredict(userVec, userIdx, fullTrain, movie, num, iufArray, amp, threshold):
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
            dist.append(pearSim(userVec, row, iufArray))
    unSignedDist = []
    for i in dist:
        unSignedDist.append(abs(i))
    kDist = topN(unSignedDist, num)
    nu = 0.0
    de = 0.0
    for i in range(len(kDist)):
        if(unSignedDist[kDist[i]] < threshold):
            continue
        # calculate trainAvg
        trainAvg = np.mean(fullTrain[kDist[i]])
        nu += dist[kDist[i]] * math.pow(unSignedDist[kDist[i]], amp-1) * (fullTrain[kDist[i]][movie] - trainAvg)
        de += math.pow(unSignedDist[kDist[i]], amp)

    if (abs(nu) < 0.00001):
        return uAvg
    return nu/de + uAvg

def itemSimDOK(uAvgArr, fullTrainDOK, ii, jj):
    nu = 0.0
    denoL = 0.0
    denoR = 0.0
    k = fullTrainDOK.shape[0]

    for i in range(k):
        nu += (fullTrainDOK[i, ii]-uAvgArr[i]) * (fullTrainDOK[i, jj]-uAvgArr[i])
        denoL += math.pow((fullTrainDOK[i, ii]-uAvgArr[i]), 2)
        denoR += math.pow((fullTrainDOK[i, jj]-uAvgArr[i]), 2)
    if denoR < 0.00001 or denoL < 0.00001:
        return 0.0
    return nu/((math.sqrt(denoL))*(math.sqrt(denoR)))


def itemBasePredict(uAvgArr, userIdx, fullTrainCSC, simMat, movie, num, iufArray, amp, threshold):

    dist = simMat[movie]

    unSignedDist = []

    for i in dist:
        unSignedDist.append(abs(i))

    kDist = topN(unSignedDist, num)
    nu = 0.0
    de = 0.0
    for i in range(len(kDist)):
        if(unSignedDist[kDist[i]] < threshold):
            continue
        nu += dist[kDist[i]] * math.pow(unSignedDist[kDist[i]], amp-1) * (fullTrainCSC[userIdx, kDist[i]] - uAvgArr[userIdx])
        de += math.pow(unSignedDist[kDist[i]], amp)


    if (abs(nu) < 0.00001):
        return uAvgArr[userIdx]
    return nu/de + uAvgArr[userIdx]

def runPredict(IUF, amp, threshold, knnNum, inputFile, outputFile, predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    allTest, testCo = save2sparse(inputFile)
    if(IUF):
        iufArray = calIUF (fullTrain)
    else:
        iufArray = [1] * 1000

    start = time.time()
    out = open(outputFile, 'w')
    for co in testCo:
        rating = predictFunc(allTest[co[0]], co[0], fullTrain, co[1], knnNum, iufArray, amp, threshold)
        rating = int(round(rating))
        if (rating > 5):
            rating = 5
        if (rating < 1):
            rating = 1
        out.write(str(co[0]) + " " + str(co[1] + 1)+ " " + str(rating) + "\n")
    end = time.time()
    print "time used to run: " , end - start

def runUserTesting(IUF, amp, threshold, knnNum, random, predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    train, test = train_test_split(fullTrain, test_size = 0.2, random_state=random)
    # create test
    test5 = []
    ans = []
    testCo = []
    for j in range(len(test)):
        tmpTest5 = []
        testNum = 0
        for i in range(1000):
            # create test with only 5 rating
            if testNum < 5:
                tmpTest5.append(test[j][i])
            else:
                tmpTest5.append(0)
            if test[j][i] > 0:
                testNum += 1
            if test[j][i] > 0 and testNum >= 5:
                # log testIdx
                testCo.append((j,i))
                # log ans
                ans.append(test[j][i])
        test5.append(tmpTest5)
    test5 = ss.csr_matrix(test5)

    ratings = []
    if(IUF):
        iufArray = calIUF (fullTrain)
    else:
        iufArray = [1] * 1000


    for co in testCo:
        rating = predictFunc(test5[co[0]], co[0], fullTrain, co[1], knnNum, iufArray, amp, threshold)
        rating = int(round(rating))
        if (rating > 5):
            rating = 5
        if (rating < 1):
            rating = 1
        ratings.append(rating)

    return mean_absolute_error(ans, ratings)

def runItemTesting(IUF, amp, threshold, knnNum, random, predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    train, test = train_test_split(fullTrain, test_size = 0.2, random_state=random)
    # create test
    test5 = []
    ans = []
    testCo = []
    for j in range(len(test)):
        tmpTest5 = []
        testNum = 0
        for i in range(1000):
            # create test with only 5 rating
            if testNum < 5:
                tmpTest5.append(test[j][i])
            else:
                tmpTest5.append(0)
            if test[j][i] > 0:
                testNum += 1
            if test[j][i] > 0 and testNum >= 5:
                # log testIdx
                testCo.append((j,i))
                # log ans
                ans.append(test[j][i])
        test5.append(tmpTest5)
    test5 = ss.csr_matrix(test5)

    ratings = []
    if(IUF):
        iufArray = calIUF (fullTrain)
    else:
        iufArray = [1] * 1000

    fullTrainCSC = ss.csc_matrix(fullTrain)
    fullTrainDOK = ss.dok_matrix(fullTrain)
    uAvgArr = []

    for i in range(fullTrainDOK.shape[0]):
        tmp = 0.0
        tmpIdx = 0
        for j in range(fullTrainDOK.shape[1]):
            if (fullTrainDOK[i, j] > 0.1):
                tmp += fullTrainDOK[i, j]
                tmpIdx += 1
        uAvgArr.append(tmp/tmpIdx)


    ############ Calculate SimMat
    # simMat = []
    # for i in range(1000):
    #     tmp = []
    #     for j in range(1000):
    #         print (i, j)
    #         if (i == j):
    #             tmp.append(0)
    #         elif (i > j):
    #             tmp.append(simMat[j][i])
    #         else:
    #             # tmp.append(itemSim(uAvgArr, fullTrainCSC.getcol(i), fullTrainCSC.getcol(j)))
    #             tmp.append(itemSimDOK(uAvgArr, fullTrainDOK, i, j))
    #     simMat.append(tmp)
    # np.savetxt("simMat.csv", simMat, delimiter=",")

    ############ read SimMat
    simMat = genfromtxt('simMat.csv', delimiter=',')

    for co in testCo:
        rating = predictFunc(uAvgArr, co[0], fullTrainCSC, simMat, co[1], knnNum, iufArray, amp, threshold)
        rating = int(round(rating))
        if (rating > 5):
            rating = 5
        if (rating < 1):
            rating = 1
        ratings.append(rating)

    return mean_absolute_error(ans, ratings)

def runStackedPredict(inputFile, outputFile, predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    allTest, testCo = save2sparse(inputFile)
    iufArray1 = calIUF (fullTrain)
    iufArray2 = [1] * 1000
    simMat = genfromtxt('simMat.csv', delimiter=',')
    fullTrainDOK = ss.dok_matrix(fullTrain)
    fullTrainCSC = ss.csc_matrix(fullTrain)
    uAvgArr = []

    for i in range(fullTrainDOK.shape[0]):
        tmp = 0.0
        tmpIdx = 0
        for j in range(fullTrainDOK.shape[1]):
            if (fullTrainDOK[i, j] > 0.1):
                tmp += fullTrainDOK[i, j]
                tmpIdx += 1
        uAvgArr.append(tmp/tmpIdx)

    start = time.time()
    out = open(outputFile, 'w')
    for co in testCo:
        rating = predictFunc(allTest[co[0]], co[0], fullTrain, co[1], 30, iufArray2, 1.5, 0.2)
        rating /= 2
        rating = int(round(rating))
        if (rating > 5):
            rating = 5
        if (rating < 1):
            rating = 1
        out.write(str(co[0]) + " " + str(co[1] + 1)+ " " + str(rating) + "\n")
    end = time.time()
    print "time used to run: " , end - start

def fullUserTest(loop, IUF, amp, threshold, knnNum, predictFunc):
    start = time.time()
    a = 0.0
    for i in range(loop):
        a += runUserTesting(IUF, amp, threshold, knnNum, i, predictFunc)
    print a/loop
    end = time.time()
    print "time used to run: " , end - start


def fullItemTest(loop, IUF, amp, threshold, knnNum, predictFunc):
    start = time.time()
    a = 0.0
    for i in range(loop):
        a += runItemTesting(IUF, amp, threshold, knnNum, i, predictFunc)
    print a/loop
    end = time.time()
    print "time used to run: " , end - start

def runAllTest():
    print "Basic Cos"
    fullUserTest(1, False, 1, 0, 160, cosPredict)
    print "Cos + IUF"
    fullUserTest(1, True, 1, 0, 160, cosPredict)
    print "Cos + Case Modification"
    fullUserTest(1, False, 2.5, 0, 160, cosPredict)
    print "Cos + IUF + Case Modification"
    fullUserTest(1, True, 2.5, 0, 160, cosPredict)

    print "Basic Pearson"
    fullUserTest(1, False, 1, 0, 160, pearPredict)
    print "Pearson + IUF"
    fullUserTest(1, True, 1, 0, 160, pearPredict)
    print "Pearson + Case Modification"
    fullUserTest(1, False, 2.5, 0, 160, pearPredict)
    print "Pearson + IUF + Case Modification"
    fullUserTest(1, True, 2.5, 0, 160, pearPredict)

    print "Item Base 10NN"
    fullItemTest(1, False, 1, 0, 10, itemBasePredict)

def runAllPrediction():
    intputList = ["test5.txt", "test10.txt", "test20.txt"]
    outputList = ["result5.txt", "result10.txt", "result20.txt"]
    for i in range(3):
        runPredict(False, 2, 0.2, 100, intputList[i], outputList[i], cosPredict)

def runAllStackedPrediction():
    intputList = ["test5.txt", "test10.txt", "test20.txt"]
    outputList = ["result5555.txt", "result10000.txt", "result20000.txt"]
    for i in range(3):
        runStackedPredict(intputList[i], outputList[i], cosPredict)

runAllStackedPrediction()
# runAllTest()
