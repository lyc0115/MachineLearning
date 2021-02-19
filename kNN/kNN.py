from numpy import *
# 导入运算符模块
import operator
# 导入os模块中的listdir函数
from os import listdir


def createDataset():
    """
    创建数据集和标签
    :return:
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, k):
    """
    K-近邻分类算法
    :param inX: 用于分类的输入向量
    :param dataset: 输入的训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return:
    """
    # 1.距离计算
    # shape读取矩阵的长度，0代表第一维长度
    dataSetSize = dataset.shape[0]
    # b = tile(a,(m,n)):即是把a数组里面的元素复制n次放进一个数组c中，然后再把数组c复制m次放进一个数组b中
    diffMat = tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 2.排序并选择与当前点距离最小的k个点
    # argsort函数返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 统计前k个点所在类别出现的频率
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    将待处理数据的格式转换为分类器可以接受的格式
    :param filename: 待处理的数据
    :return: returnMat  训练样本矩阵   classLabelVector  类标签向量
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征值
    :param dataSet:
    :return:
    """
    # 返回每一列的最小值
    minVals = dataSet.min(0)
    # 返回每一列的最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    验证程序分类效果
    :return:
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))


def classifyPerson():
    reasultList = ['not at all', 'in small doses', 'in laege doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals/ranges), normMat, datingLabels, 3)
    print("You will probably like this person: ", reasultList[classifierResult - 1])


def img2vector(filename):
    """
    将一个32*32的二进制图像矩阵转换为1*1024的向量
    :param filename:
    :return:
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    """
    手写数字识别
    :return:
    """
    hwLables = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        hwLables.append(classNumstr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLables, 3)
        print("the classifier came back with %d, the real answer is: %d" % (classifierResult, classNumstr))
        if (classifierResult != classNumstr):
            errorCount += 1.0
    print("\nthe total number of errors is %d" % errorCount)
    print("\nthe total error rate is %f" % (errorCount / float(mTest)))
