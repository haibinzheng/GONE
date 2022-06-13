import sys
sys.dont_write_bytecode = True

import numpy as np

import pickle
from sklearn.model_selection import train_test_split
import random
import lasagne
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import argparse
import deeplearning as dp
import classifier

def preprocessingNews(toTrainData, toTestData):
	def normalizeData(X):
		offset = np.mean(X, 0)
		scale = np.std(X, 0).clip(min=1)
		X = (X - offset) / scale
		X = X.astype(np.float32)
		return X
	return normalizeData(toTrainData),normalizeData(toTestData)


def shuffleAndSplitData(dataX, dataY, cluster):
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)
    toTrainData = np.array(dataX[:cluster])
    toTrainLabel = np.array(dataY[:cluster])

    shadowData = np.array(dataX[cluster:cluster * 2])
    shadowLabel = np.array(dataY[cluster:cluster * 2])

    toTestData = np.array(dataX[cluster * 2:cluster * 3])
    toTestLabel = np.array(dataY[cluster * 2:cluster * 3])

    shadowTestData = np.array(dataX[cluster * 3:cluster * 4])
    shadowTestLabel = np.array(dataY[cluster * 3:cluster * 4])

    return toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel


def initializeData(dataset, orginialDatasetPath, dataFolderPath='./data/'):

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    X = np.concatenate((newsgroups_train.data, newsgroups_test.data), axis=0)
    y = np.concatenate((newsgroups_train.target, newsgroups_test.target), axis=0)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    X = X.toarray()
    print("Preprocessing data")
    print(X.shape)
    cluster = 4500
    dataPath = dataFolderPath + dataset + '/Preprocessed'
    toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
            X, y, cluster)
    toTrainDataSave, toTestDataSave = preprocessingNews(toTrainData, toTestData)
    shadowDataSave, shadowTestDataSave = preprocessingNews(shadowData, shadowTestData)

    try:
        os.makedirs(dataPath)
    except OSError:
        pass

    np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
    np.savez(dataPath + '/targetTest.npz', toTestDataSave, toTestLabel)
    np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
    np.savez(dataPath + '/shadowTest.npz', shadowTestDataSave, shadowTestLabel)

    print("Preprocessing finished\n\n")

# initializeData('News', 1)
targetTrain = np.load('/home/NewDisk/sgwc/attack/ML-Leaks-master/data/News/Preprocessed/targetTrain.npz')
targetTest = np.load('/home/NewDisk/sgwc/attack/ML-Leaks-master/data/News/Preprocessed/targetTest.npz')
shadowTrain = np.load('/home/NewDisk/sgwc/attack/ML-Leaks-master/data/News/Preprocessed/shadowTrain.npz')
shadowTest = np.load('/home/NewDisk/sgwc/attack/ML-Leaks-master/data/News/Preprocessed/shadowTest.npz')
x = targetTrain.files
target_train_x = targetTrain['arr_0']
target_train_y = targetTrain['arr_1']
a= [1]