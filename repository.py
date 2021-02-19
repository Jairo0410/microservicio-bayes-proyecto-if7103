from pandas import read_csv
from os import path

current_path = path.dirname(path.realpath(__file__))
data_dir = path.join(current_path, 'data')
probs_dir = path.join(current_path, 'probs')

__atractivos_data = read_csv(path.join(data_dir, 'atractivos.csv'))

def formProbsFilePath(modelName, feature, targetClass):
    return path.join(probs_dir, modelName + '_'+ 'probabilities' + '_' + feature + '_' + targetClass + '.csv')

def formFreqFilePath(modelName, feature, targetClass):
    return path.join(probs_dir, modelName + '_' + 'frequencies' + '_' + feature + '_' + targetClass + '.csv')

def saveProbabilites(dataframe, modelName, feature, targetClass):
    dataframe.to_csv(formProbsFilePath(modelName, feature, targetClass))

def saveFrequences(dataframe, modelName, feature, targetClass):
    dataframe.to_csv(formFreqFilePath(modelName, feature, targetClass))

def readProbabilities(modelName, feature, targetClass):
    return read_csv(formProbsFilePath(modelName, feature, targetClass))

def getAtractivosData():
    return __atractivos_data
