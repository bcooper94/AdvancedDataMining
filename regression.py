
# coding: utf-8

# In[820]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.linear_model import Lasso
from IPython.display import display, HTML



# ## Convert Categorical Data to Dummy Variables

# In[821]:

def categoricalColumnNames(columnName, attributeList):
    return ['{}_{}'.format(columnName, attribute)
            for attribute in attributeList]


# def categoricalVectors(attributeList):
#     return [np.zeros(len(attrValues) - 1) for attr in attributeList]
    

# Convert nominal variable x to its dummy vector variable representation
def _convertNominalVariable(dataFrame, columnName):
    attrValues = dataFrame[columnName].unique()
    newColumns = categoricalColumnNames(columnName, attrValues)
    encodingList = [np.zeros(len(attrValues) - 1) for attr in attrValues]
    
    for newColumn, value in zip(newColumns, attrValues):
        dataFrame[newColumn] = pd.Series(dtype=np.int64)
        dataFrame.loc[dataFrame[columnName] == value, newColumn] = int(1)
        dataFrame.loc[dataFrame[columnName] != value, newColumn] = int(0)
    
    for i in range(1, len(attrValues)):
        encodedAttr = encodingList[i]
        encodedAttr[i - 1] = int(1)
    
    return dataFrame


def convertNominalVariables(dataFrame, columnList):
    newDf = pd.DataFrame(dataFrame)
    
    for attribute in columnList:
        newDf = _convertNominalVariable(newDf, attribute)
    return newDf
        
    
# Convert ordinal variable x to its dummy vector variable representation
def _convertOrdinalVariable(dataFrame, columnName, orderedAttributes):
    oneHotDf = _convertNominalVariable(dataFrame, columnName)
    newAttrNames = categoricalColumnNames(columnName, orderedAttributes)
    attrIndices = {}
    
    for i, attr in enumerate(orderedAttributes):
        attrIndices[attr] = int(i)
    
    for i, newOldAttr in enumerate(zip(orderedAttributes, newAttrNames)):
        oldAttr, newAttr = newOldAttr
        
        for attrName in newAttrNames[:i]:
            oneHotDf.loc[oneHotDf[columnName] == oldAttr, attrName] = int(1)
    return oneHotDf


# ## Evaluation Measures

# In[822]:

def checkR2(x, y):
    slope, intercept, r2, p_val, std_err = stats.linregress(x, y)
    print('slope={}, intercept={}, R2={}'.format(slope, intercept, r2 ** 2))
    
# Returns 2 measures of R2: (1 - SSE / SST, SSR / SST)
def computeR2(x, y, beta):
    yMean = np.mean(y)
#     print('Y:', y)
#     print('Y mean:', yMean)
    yPredicted = np.array([linRegEstimate(xVal, beta) for xVal in x])
#     print('Y predicted:', yPredicted)
    ssTotal = np.sum(np.square(y - yMean))
    ssRegression = np.sum(np.square(yPredicted - yMean))
    ssError = np.sum(np.square(y - yPredicted))
#     print('SST={}, SSR={}, SSE={}'.format(ssTotal, ssRegression, ssError))
    
    return 1 - ssError / ssRegression

"""
Computes the accuracy, precision, recall, and F-score of a dataframe
Params:
    testDf: pandas.DataFrame
    predictedDf: pandas.DataFrame
Returns: (accuracy, precision, recall, F-score)
"""
def scoreLogClassification(testVals, predictedVals):
    tp = tn = fp = fn = 0
    
    for testVal, predictedVal in zip(testVals, predictedVals):
        if testVal == predictedVal:
            if testVal == 1:
                tp += 1
            else:
                tn += 1
        else:
            if testVal == 1:
                fn += 1
            else:
                fp += 1
    
    accuracy = (tp + tn) / len(testVals)
    precision = tp / (tp + fp) if tp + fp > 0 else float('NaN')
    recall = tp / (tp + fn) if tp + fn > 0 else float('NaN')
    fScore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float('NaN')
    
    return accuracy, precision, recall, fScore


# ## Linear Regression Functions

# In[823]:

"""
inputVals: vector in form of [x1, x2, ..., xn]
beta: vector of beta coefficients in form [b0, b1, b2, ..., bn]
"""
def linRegEstimate(inputVals, beta):
    inputMatrix = np.insert(inputVals, 0, 1)
    return np.sum(inputMatrix * beta)


"""
matrix: matrix of points in form [[x1, x2, ..., xn], ...]
returns: (beta, pearsonCorrelation) tuple
"""
def computeLinRegBeta(x, y):
    # Reshape 1-D arrays to column format
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
        
    m = len(x)
    newX = np.insert(x, 0, np.ones(m), axis=1)
    
    # Reshape y from vector to 1-D matrix rows
    newY = y.reshape(-1, 1)
#     print('Y:', newY)
    xTxInverse = np.linalg.inv(newX.T.dot(newX))
    beta = xTxInverse.dot(newX.T).dot(newY)
    
    return beta.T[0], np.corrcoef(x.T, newY.T[0])[0, 1] ** 2


# # Logistic Regression

# In[824]:

"""
Returns a function that takes a vector X (same dimensions as parameter x),
    and predicts class 0 or 1 based on X.
    Return type: (function, R^2 correlation)
Params:
    x: input vector
    y: 1-D class vector of 0s and 1s
"""
def getLogisticRegFunc(x, y):
    beta, correlation = computeLinRegBeta(x, y)
    def logRegEstimate(xVector):
        estimate = 1 / (1 + np.exp(linRegEstimate(xVector, beta)))
        return estimate, int(round(estimate))
    return logRegEstimate, correlation


# # Utilities

# In[825]:

def plotSingleVarRegressions(xCols, yCol):
    yColValues = yCol.as_matrix()
    
    for col in xCols:
        colValues = col.as_matrix()
        colBetas, colCorrelation = computeLinRegBeta(colValues, yColValues)

        xMin = colValues.min()
        xMax = colValues.max()
        xRange = np.arange(xMin * .85, xMax * 1.15, (xMax - xMin) / 10)
        predictedLine = [linRegEstimate([x], colBetas) for x in xRange]

        fig = plt.figure()
        graph = fig.add_subplot(111)
        fig.suptitle('Predicting {} levels from {} levels'.format(yCol.name, col.name),
                    y=1.08, fontsize=14, fontweight='bold')
        graph.set_title('Beta: {}, R2: {}'.format(colBetas, colCorrelation))
        graph.title.set_position([0.5, 1.05])
        graph.set_ylabel(yCol.name)
        graph.set_xlabel(col.name)
        graph.plot(colValues, yColValues, 'bo', xRange, predictedLine, 'r-', linewidth=3.0)
        
def getDfNamesByType(df):
    columnNamesByType = df.columns.to_series().groupby(df.dtypes).groups
    return {key.name: val for key, val in columnNamesByType.items()}

def displayBetaTable(attrNames, beta):
    betaAttrMap = {attr: beta for attr, beta in zip(['Intercept'] + attrNames, beta)}
    display(pd.DataFrame(betaAttrMap, index=np.arange(1)))
