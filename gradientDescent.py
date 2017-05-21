from sys import maxsize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from regression import getDfNamesByType, linRegEstimate


# Stochastic gradient descent
def SGD(x, y, learningRate, error):
    x = np.array(x)
    y = np.array(y)

    # Reshape 1-D arrays to column format
    # if len(x.shape) == 1:
    #     x = x.reshape(-1, 1)

    N = len(x)
    currentError = maxsize
    lastError = 0
    beta = np.zeros(x.shape[1])

    while currentError > error:
        yPredicted = x.dot(beta)
        sqErrGradient = np.array(np.dot(x.T, (yPredicted - y)) / N)
        beta -= learningRate * sqErrGradient
        currentError = np.sum(np.square(y - yPredicted)) / N
        # print('Error: {}, SE Gradient: {}'.format(currentError, sqErrGradient))

        if abs(lastError - currentError) < error:
            break
        lastError = currentError

    return beta


def main():
    convertEuropeanFloats = lambda val: float(val.replace(',', '.'))
    airQualityDf = pd.read_csv('data/AirQualityUCI.csv', sep=';',
                               converters={
                                   'CO(GT)': convertEuropeanFloats,
                                   'C6H6(GT)': convertEuropeanFloats,
                                   'T': convertEuropeanFloats,
                                   'RH': convertEuropeanFloats
                               })
    airQualityDf.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)

    columnNamesByType = getDfNamesByType(airQualityDf)
    airQualityXColNames = [col for col in list(columnNamesByType['float64']) + list(columnNamesByType['int64'])
                           if col != 'CO(GT)']
    airQualityDf.drop(airQualityDf[airQualityDf['CO(GT)'] == -200.0].index, inplace=True)

    for xCol in airQualityXColNames:
        # -200 means missing value
        airQualityDf.drop(airQualityDf[airQualityDf[xCol] == -200.0].index, inplace=True)

    airQualityY = airQualityDf['CO(GT)']
    airQualityXCols = [airQualityDf[attr]
                       for attr in airQualityXColNames]

    COValues = airQualityY.as_matrix()
    # plotSingleVarRegressions(airQualityXCols, airQualityY)

    c6h6GTValues = airQualityDf['C6H6(GT)'].as_matrix()
    c6h6GTX = np.insert(c6h6GTValues.reshape(-1, 1), 0, np.ones(len(c6h6GTValues)), axis=1)
    print(c6h6GTX)
    # c6h6GTValues = np.insert(c6h6GTValues, 0, np.ones(len(c6h6GTValues)), axis=1)
    # c6h6VsCOBetas, c6h6VsCOCorrelation = computeLinRegBeta(c6h6GTValues, COValues)
    c6h6VsCOBetas = SGD(c6h6GTX, COValues, 0.001, 0.0001)
    print('Single regression beta:', c6h6VsCOBetas)
    # displayBetaTable(airQualityXColNames, c6h6VsCOBetas)
    # print('C6H6(GT) to predict CO(GT) R^2:', c6h6VsCOCorrelation)
    # # print('C6H6(GT) to CO(GT) beta:', c6h6VsCOBetas)
    #
    xRange = np.arange(0, 70, 5)
    predictedLine = [linRegEstimate([c6h6], c6h6VsCOBetas) for c6h6 in xRange]

    plt.plot(c6h6GTValues, COValues, 'bo', xRange, predictedLine, 'r-', linewidth=3.0)
    plt.title('Predicting CO content from C6H6 content')
    plt.xlabel('C6H6 (microgram/m^3)')
    plt.ylabel('CO (mg/m^3)')
    plt.show()

    benzenePT08Df = airQualityDf[['C6H6(GT)', 'PT08.S2(NMHC)']]
    benzenePT08 = benzenePT08Df.as_matrix()
    benzenePT08X = np.insert(benzenePT08, 0, np.ones(len(benzenePT08)), axis=1)
    print('BenzenePT08:', benzenePT08)
    benzenePT08Betas = SGD(benzenePT08X, COValues, 0.0000001, 0.00001)
    print('Benzene-PT08 Beta:', benzenePT08Betas)

    benzeneMin = benzenePT08[:, 0].min()
    benzeneMax = benzenePT08[:, 0].max()
    PT08Min = benzenePT08[:, 1].min()
    PT08Max = benzenePT08[:, 1].max()

    benzeneDomain = np.arange(benzeneMin, benzeneMax, (benzeneMax - benzeneMin) / 10)
    PT08Domain = np.arange(PT08Min, PT08Max, (PT08Max - PT08Min) / 10)
    benzenePT08Domains = np.array([benzeneDomain, PT08Domain]).T
    benzenePT08PredictedCO = [linRegEstimate(xy, benzenePT08Betas)
                              for xy in benzenePT08Domains]

    fig = plt.figure()
    fig.suptitle('Benzene and Titania to Predict CO Levels',
                 y=1.05, fontsize=14, fontweight='bold')

    benzPT08Plot = fig.add_subplot(111, projection='3d')
    benzPT08Plot.scatter(benzenePT08[:, 0], benzenePT08[:, 1], COValues)
    gridBenzPTO8, gridPT08Domain = np.meshgrid(benzeneDomain, PT08Domain)
    benzPT08Plot.plot_surface(gridBenzPTO8, gridPT08Domain, benzenePT08PredictedCO, color='r')
    benzPT08Plot.set_title('Beta: {}, R2: {}'.format(benzenePT08Betas, 'Not implemented'),
                           y=1.05, fontsize=10)
    benzPT08Plot.set_xlabel('C6HG (microgram/m^3)', labelpad=10.5)
    benzPT08Plot.set_ylabel('PT08.S2', labelpad=10.5)
    benzPT08Plot.set_zlabel('CO mg/m^3')

    benzPT08Plot.set_yticks(np.arange(200, 2000, 400))
    benzPT08Plot.dist = 12

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
