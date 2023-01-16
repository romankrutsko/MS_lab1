import math
import random
from matplotlib import pyplot as plt
import numpy as np
import statistics as st
import scipy.integrate as integrate

hi_squared_table = {
    1: 3.8,
    2: 6.0,
    3: 7.8,
    4: 9.5,
    5: 11.1,
    6: 12.6,
    7: 14.1,
    8: 15.5,
    9: 16.9,
    10: 18.3,
    11: 19.7,
    12: 21.0,
    13: 22.4,
    14: 23.7,
    15: 25.0,
    16: 26.3,
    17: 27.6,
    18: 28.9,
    19: 30.1,
    20: 31.4,
    21: 32.7,
    22: 33.9,
    23: 35.2,
    24: 36.4,
    25: 37.7,
    26: 38.9,
    27: 40.1,
    28: 41.3,
    29: 42.6,
    30: 43.8
}


def drawGraph(xArray: list, space=0):
    plt.xlim([min(xArray) - space, max(xArray) + space])

    bins = np.arange(-100, 100, 0.5)
    plt.hist(xArray, bins=bins, alpha=0.5)
    plt.title('exponent function')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
    return


def drawFunction(xArray: list, yArray: list):
    plt.scatter(xArray, yArray)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
    return


def exponDistr(lamda: float) -> list:
    xArray = []
    for item in range(0, 10000):
        randomValue = random.random()
        xArray.append(-np.log(randomValue) / lamda)

    return (xArray)


def expDistRule(lamda: float, xArray: list) -> list:
    yArray = []
    for item in xArray:
        yArray.append(1 - math.pow(math.e, -lamda * item))
    return (yArray)


def normalDistr(a: float, sigma: float) -> list:
    xArray = []
    for itemFirst in range(0, 10000):
        mu = -6
        for item in range(0, 12):
            mu += random.random()
        xArray.append(sigma * mu + a)
    return (xArray)


def normalDistrSecond(a: float, sigma: float, xArray: list) -> list:
    yArray = []
    for item in xArray:
        yArray.append(1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-math.pow(item - a, 2) / (2 * sigma * sigma)))
    return yArray


def evenDistr(a=math.pow(5, 12), c=math.pow(2, 9)) -> list:
    z = a * random.random() % c
    xArray = []
    for item in range(0, 10000):
        z = a * z % c
        xArray.append(z / c)

    return xArray


def getInterv(array, count) -> list:
    intervalSize = (max(array) - min(array)) / count
    intervals = []

    counter = min(array)
    for item in range(0, count):
        intervals.append([0, [counter, counter + intervalSize]])
        counter = counter + intervalSize
    return (intervals)


def getActualInterv(array, intervals) -> list:
    for item in array:
        for interval in intervals:
            if item > interval[1][0] and item <= interval[1][1]:
                interval[0] = interval[0] + 1
    return intervals


def getExpExpValue(first, second, lamda):
    return np.exp(-lamda * first) - np.exp(-lamda * second)


def getExpNormValue(first, second, alpha, sygma):
    def func(x):
        return 1 / (sygma * np.sqrt(2 * np.pi)) * np.exp(- (x - alpha) ** 2 / (2 * sygma ** 2))

    res, mes = integrate.quad(func, first, second)
    return res


def getExpEvenValue(first, second, array):
    return (second - first) / (max(array) - min(array))


def getChiValues(expectedList, observedList, intervals):
    obsrvedChiSqr = 0
    for i in range(len(intervals)):
        expectedValue = 10000 * expectedList[i]
        obsrvedChiSqr += pow(observedList[i][0] - expectedValue, 2) / expectedValue
    return obsrvedChiSqr, hi_squared_table[len(intervals) - 1]


# first
xArray = exponDistr(0.5)
intervals = getInterv(xArray, 20)
intervalsWithValues = getActualInterv(xArray, getInterv(xArray, 20))

print("mean = " + str(st.mean(xArray)))
print("pvariance = " + str(st.pvariance(xArray)))
print(*intervalsWithValues, sep="\n")

xExpArray = []
for item in intervals:
    xExpArray.append(getExpExpValue(item[1][0], item[1][1], 0.5))

ChiValue = getChiValues(xExpArray, intervalsWithValues, intervals)
print("Observed X^2: " + str(ChiValue[0]))
print("Expected X^2: " + str(ChiValue[1]))

drawGraph(xArray)
drawFunction(xArray, expDistRule(0.5, xArray))

# second
alpha = 9
sigma = 5
xArrayNormal = normalDistr(alpha, sigma)
intervals = getInterv(xArrayNormal, 20)
intervalsWithValues = getActualInterv(xArrayNormal, intervals)

print("mean = " + str(st.mean(xArrayNormal)))
print("pvariance = " + str(st.pvariance(xArrayNormal)))
print(*intervalsWithValues, sep="\n")

xExpArrayNorm = []
for item in intervals:
    xExpArrayNorm.append(getExpNormValue(item[1][0], item[1][1], alpha, sigma))

ChiValue = getChiValues(xExpArrayNorm, intervalsWithValues, intervals)

print("Observed X^2: " + str(ChiValue[0]))
print("Expected X^2: " + str(ChiValue[1]))

drawGraph(xArrayNormal)
drawFunction(xArrayNormal, normalDistrSecond(alpha, sigma, xArrayNormal))

# third
xArrayEven = evenDistr()
intervals = getInterv(xArrayEven, 10)
intervalsWithValues = getActualInterv(xArrayEven, intervals)

print("mean = " + str(st.mean(xArrayEven)))
print("pvariance = " + str(st.pvariance(xArrayEven)))
print(*intervalsWithValues, sep="\n")

xExpArrayRivn = []
for item in intervals:
    xExpArrayRivn.append(getExpEvenValue(item[1][0], item[1][1], xArrayEven))

ChiValue = getChiValues(xExpArrayRivn, intervalsWithValues, intervals)

print("Observed X^2: " + str(ChiValue[0]))
print("Expected X^2: " + str(ChiValue[1]))

drawGraph(xArrayEven, 0.1)