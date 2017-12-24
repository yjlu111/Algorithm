# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 21:39:14 2017

@author: 卢友军
"""
#3.1 收集数据
import urllib
import json

def geoGrab(stAddress,city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'  #返回JSON
    params['appid'] = 'ppp68N8t' #注册，创建桌面后会获得一个appid
    params['location'] = '%s %s %(stAddress,city)'
    url_params = urllib.urlencode(params)  # 将params字典转换为可以通过URL进行传递的字符串格式
    yahooApi = apiStem+url_params
    print(yahooApi) # 输出URL
    c = urllib.ulropen(yahooApi) #读取返回值
    return json.loads(c.read()) # 返回一个字典
from time import sleep
def massPlaceFind(fileName):
    fw = open('place.txt','w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr =line.split('\t') #是以tab分隔的文本文件
        retDict = geoGrab(lineArr[1],lineArr[2]) #读取2列和第3列
        if retDict['ResultSet']['Error']==0:    # 检查输出字典，判断有没有出错
            lat = float(retDict['ResultSet']['Results'][0]['latitude']) #读取经纬度
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f"%(lineArr[0],lat,lng))
            fw.write('%s\t%f\t%f\n' % (line,lat,lng))  #添加到对应的行上
        else: print("error fetching") #有错误时不需要抽取经纬度
        sleep(1) #避免频繁调用API，过于频繁的话请求会被封掉
        fw.close()

#3.2对地理坐标进行聚类
import numpy as np
# K-均值聚类支持函数
def loadDataSet(fileName):      
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) 
        dataMat.append(fltLine)
    return dataMat
# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA-vecB, 2))) 
# 为给定数据集构建一个包含k个随机质心的集合,是以每列的形式生成的
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n))) 
    for j in range(n):  
        minJ = min(dataSet[:,j])  # 找到每一维的最小
        rangeJ = float(max(dataSet[:,j]) - minJ) # 每一维的最大和最小值之差
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1)) # 生成随机值
        #print centroids[:,j]
    return centroids  # 返回随机质心,是和数据点相同的结构
# k--均值聚类算法(计算质心--分配--重新计算)
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent): # k是簇的数目
    m = np.shape(dataSet)[0]  # 得到样本的数目
    clusterAssment = np.mat(np.zeros((m,2))) #  创建矩阵来存储每个点的簇分配结果
                                       #  第一列：记录簇索引值，第二列：存储误差，欧式距离的平方
    centroids = createCent(dataSet, k)  # 创建k个随机质心
    clusterChanged = True
    while clusterChanged:  # 迭代使用while循环来实现
        clusterChanged = False  
        for i in range(m):  # 遍历每个数据点，找到距离每个点最近的质心
            minDist = np.inf; minIndex = -1
            for j in range(k):  # 寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: # 更新停止的条件
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 # minDist**2就去掉了根号         
        for cent in range(k):  # 更新质心的位置
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]] 
            centroids[cent,:] = np.mean(ptsInClust, axis=0) # 然后计算均值，axis=0:沿列方向 
    #print 'centroids:',centroids
    return centroids, clusterAssment # 返回簇和每个簇的误差值，误差值是当前点到该簇的质心的距离
    # 二分k--均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0] 
    clusterAssment = np.mat(np.zeros((m,2))) # 存储数据集中每个点的簇分配结果及平方误差
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] # 计算整个数据集的质心：1*2的向量
    centList =[centroid0] # []的意思是使用一个列表保存所有的质心,簇列表,[]的作用很大
    for j in range(m):  # 遍历所有的数据点，计算到初始质心的误差值，存储在第1列
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):  # 不断对簇进行划分，直到k
        lowestSSE = np.inf  # 初始化SSE为无穷大
        for i in range(len(centList)): # 遍历每一个簇
            #print 'i:',i               # 数组过滤得到所有的类别簇等于i的数据集
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            # 得到2个簇和每个簇的误差，centroidMat：簇矩阵  splitClustAss：[索引值,误差]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # centroidMat是矩阵
            sseSplit = sum(splitClustAss[:,1])  # 求二分k划分后所有数据点的误差和     
                                             # 数组过滤得到整个数据点集的簇中不等于i的点集
            #print nonzero(clusterAssment[:,0].A!=i)[0]
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])# 所有剩余数据集的误差之和
            #print "sseSplit and notSplit: ",sseSplit,',',sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE: # 划分后的误差和小于当前的误差，本次划分被保存
                #print 'here..........'
                bestCentToSplit = i  # i代表簇数
                bestNewCents = centroidMat  # 保存簇矩阵
                #print 'bestNewCents',bestNewCents
                bestClustAss = splitClustAss.copy() # 拷贝所有数据点的簇索引和误差
                lowestSSE = sseSplit + sseNotSplit  # 保存当前误差和
        # centList是原划分的簇向量，bestCentToSplit是i值
        #print 'len(centList) and  bestCentToSplit ',len(centList),',',bestCentToSplit
                  # 数组过滤得到的是新划分的簇类别是1的数据集的类别簇重新划为新的类别值为最大的类别数
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 
                  # 数组过滤得到的是新划分的簇类别是0的数据集的类别簇重新划为新的类别值为i
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        #print 'the bestCentToSplit is: ',bestCentToSplit   # 代表的是划分的簇个数-1
        #print 'the len of bestClustAss is: ', len(bestClustAss) # 数据簇的数据点个数
                                   # 新划分簇矩阵的第0簇向量新增到当前的簇列表中
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] 
        #print 'centList[bestCentToSplit]:',centList[bestCentToSplit]
                        # 新划分簇矩阵的第1簇向量添加到当前的簇列表中
        centList.append(bestNewCents[1,:].tolist()[0]) # centList是列表的格式
        #print 'centList',centList
                    # 数组过滤得到所有数据集中簇类别是新簇的数据点
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return np.mat(centList), clusterAssment # 返回质心列表和簇分配结果


# 球面距离计算，这里是利用球面余弦定理
def distSLC(vecA, vecB):  # 经度和纬度用角度作为单位，这里用角度除以180然后乘以pi作为余弦函数的输入
    a = np.sin(vecA[0,1]*np.pi/180) * np.sin(vecB[0,1]*np.pi/180) 
    b = np.cos(vecA[0,1]*np.pi/180) * np.cos(vecB[0,1]*np.pi/180) * \
                      np.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0  # 返回地球表面两点之间的距离


import matplotlib.pyplot as plt
# 及簇绘图函数 
def clusterClubs(numClust=5):  # 希望分得的簇数
    datList = []  # 创建一个空列表
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])  # 对应的是纬度和经度
    datMat = np.mat(datList) # 创建一个矩阵
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure() # 创建一幅图
    rect=[0.1,0.1,0.8,0.8] # 创建一个矩形来决定绘制图的哪一部分
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<'] # 构建一个标记形状的列表来绘制散点图
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)  # 创建一个子图
    imgP = plt.imread('Portland.png')  # imread()函数基于一幅图像来创建矩阵
    ax0.imshow(imgP) # imshow()绘制该矩阵
    ax1=fig.add_axes(rect, label='ax1', frameon=False) # 在同一张图上又创建一个字图
    for i in range(numClust): # 遍历每一个簇
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
 # 主函数
clusterClubs(5)









