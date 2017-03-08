import time
from numpy import genfromtxt
import pandas as pd
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

def cosSim(userVec, row):
    uRows,uCols = userVec.nonzero()
    nu = 0.0
    denoL = 0.0
    denoR = 0.0
    for i in uCols:
        nu += userVec[uRows[0], i] * row[i]
        denoL += math.pow(userVec[uRows[0], i], 2)
        denoR += math.pow(row[i], 2)
    if (denoR < 0.00001):
        return 0.0
    return nu/((math.sqrt(denoL))*(math.sqrt(denoR)))

def calIUF(trainMat):
    iufArray = []
    trainMat = ss.csr_matrix(trainMat)
    for i in range(trainMat.shape[1]):
        oneIUF = trainMat.getcol(i).count_nonzero()
        if (oneIUF == 0):
            iufArray.append(0)
            continue
        oneIUF = math.log10(200.0/oneIUF)
        iufArray.append(oneIUF)
    return iufArray

def cosPredict(userVec, fullTrain, movie, num, IUF, iufArray):
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
        if(dist[kDist[i]] < 0.2):
            continue
        # print dist[kDist[i]]
        if(IUF):
            nu += dist[kDist[i]] * fullTrain[kDist[i]][movie] * iufArray [movie]
            de += dist[kDist[i]]
        else:
            nu += dist[kDist[i]] * fullTrain[kDist[i]][movie]
            de += dist[kDist[i]]
    if (nu < 0.00001):
        # print "guessing"
        return 3.0
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

def pearPredict(userVec, fullTrain, movie, num, IUF, iufArray):
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
        if(unSignedDist[kDist[i]] < 0.2):
            continue
        # calculate trainAvg
        trainAvg = np.mean(fullTrain[kDist[i]])
        if(IUF):
            nu += dist[kDist[i]] * (fullTrain[kDist[i]][movie] - trainAvg) * iufArray [movie]
            de += unSignedDist[kDist[i]] * iufArray [movie]
        else:
            nu += dist[kDist[i]] * (fullTrain[kDist[i]][movie] - trainAvg)
            de += unSignedDist[kDist[i]]

    if (abs(nu) < 0.00001):
        return uAvg
    return nu/de + uAvg

def runPredict(inputFile, outputFile, predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    # fullTrain = ss.csr_matrix(fullTrain)
    allTest, testCo = save2sparse(inputFile)
    start = time.time()
    out = open(outputFile, 'w')
    for co in testCo:
        # print co[0]
        rating = predictFunc(allTest[co[0]], fullTrain, co[1], 50)
        rating = int(round(rating))
        if (rating > 5):
            rating = 5
        if (rating < 1):
            rating = 1
        out.write(str(co[0]) + " " + str(co[1] + 1)+ " " + str(rating) + "\n")
    end = time.time()
    print "time used to run: " , end - start

def runTesting(IUF, predictFunc):
    fullTrain = genfromtxt('train.txt', delimiter='\t')
    # df = pd.DataFrame(fullTrain)
    train, test = train_test_split(fullTrain, test_size = 0.2, random_state=13)
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
    iufArray = calIUF (fullTrain)
    for co in testCo:
        # print co[0]
        rating = predictFunc(test5[co[0]], train, co[1], 100, IUF, iufArray)
        rating = int(round(rating))
        if (rating > 5):
            rating = 5
        if (rating < 1):
            rating = 1
        ratings.append(rating)

    print mean_absolute_error(ans, ratings)

runTesting(False, pearPredict)
runTesting(True, pearPredict)

# intputList = ["test5.txt", "test10.txt", "test20.txt"]
# outputList = ["result5.txt", "result10.txt", "result20.txt"]
# for i in range(3):
#     runPredict(intputList[i], outputList[i], cosPredict)
