import math
import random
from matplotlib import pyplot as plt
import numpy as np
import scipy.integrate as integrate

chi_squared_table = {
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

def draw_histogram(data, space=0):
    plt.xlim([min(data) - space, max(data) + space])

    bins = np.arange(-100, 100, 0.5)
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('exponent function')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def draw_scatter_plot(x, y):
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def exponential_distribution(rate: float) -> list:
    data = []
    for _ in range(0, 10000):
        random_value = random.random()
        data.append(-np.log(random_value) / rate)

    return data

def exponential_distribution_rule(rate: float, data: list) -> list:
    y = []
    for x in data:
        y.append(1 - math.pow(math.e, -rate * x))
    return y

def normal_distribution(mean: float, std_dev: float) -> list:
    data = []
    for _ in range(0, 10000):
        mu = -6
        for _ in range(0, 12):
            mu += random.random()
        data.append(std_dev * mu + mean)
    return data

def normal_distribution_rule(mean: float, std_dev: float, data: list) -> list:
    y = []
    for x in data:
        y.append(1 / (std_dev * math.sqrt(2 * math.pi)) * math.exp(-math.pow(x - mean, 2) / (2 * std_dev * std_dev)))
    return y

def uniform_distribution(a=math.pow(5, 12), c=math.pow(2, 9)) -> list:
    z = a * random.random() % c
    data = []
    for _ in range(0, 10000):
        z = a * z % c
        data.append(z / c)

    return data

def get_interval(data, count) -> list:
    interval_size = (max(data) - min(data)) / count
    intervals = []

    counter = min(data)
    for _ in range(0, count):
        intervals.append([0, [counter, counter + interval_size]])
        counter = counter + interval_size
    return intervals

def get_actual_interval(data, intervals) -> list:
    for x in data:
        for interval in intervals:
            if x > interval[1][0] and x <= interval[1][1]:
                interval[0] += 1
    return intervals


def get_exp_exp_value(first, second, lamda):
    return np.exp(-lamda * first) - np.exp(-lamda * second)

def get_exp_norm_value(first, second, mean, std_dev):
    def func(x):
        return 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std_dev ** 2))

    result, _ = integrate.quad(func, first, second)
    return result

def get_exp_uniform_value(first, second, data):
    return (second - first) / (max(data) - min(data))

def get_chi_values(expected_list, observed_list, intervals):
    observed_chi_sqr = 0
    for i in range(len(intervals)):
        expected_value = 10000 * expected_list[i]
        observed_chi_sqr += pow(observed_list[i][0] - expected_value, 2) / expected_value
    return observed_chi_sqr, chi_squared_table[len(intervals) - 1]

# first
x_array = exponential_distribution(0.5)
intervals = get_interval(x_array, 20)
intervals_with_values = get_actual_interval(x_array, get_interval(x_array, 20))

print("mean = " + str(np.mean(x_array)))
print("pvariance = " + str(np.var(x_array)))
print(*intervals_with_values, sep="\n")

x_exp_array = []
for item in intervals:
    x_exp_array.append(get_exp_exp_value(item[1][0], item[1][1], 0.5))

chi_value = get_chi_values(x_exp_array, intervals_with_values, intervals)
print("Observed Chi-squared: " + str(chi_value[0]))
print("Expected Chi-squared: " + str(chi_value[1]))

draw_histogram(x_array)
draw_scatter_plot(x_array, exponential_distribution_rule(0.5, x_array))

# second
mean = 9
std_dev = 5
x_array_normal = normal_distribution(mean, std_dev)
intervals = get_interval(x_array_normal, 20)
intervals_with_values = get_actual_interval(x_array_normal, intervals)

print("mean = " + str(np.mean(x_array_normal)))
print("pvariance = " + str(np.var(x_array_normal)))
print(*intervals_with_values, sep="\n")

x_exp_array_norm = []
for item in intervals:
    x_exp_array_norm.append(get_exp_norm_value(item[1][0], item[1][1], mean, std_dev))

chi_value = get_chi_values(x_exp_array_norm, intervals_with_values, intervals)

print("Observed Chi-squared: " + str(chi_value[0]))
print("Expected Chi-squared: " + str(chi_value[1]))

draw_histogram(x_array_normal)
draw_scatter_plot(x_array_normal, normal_distribution_rule(mean, std_dev, x_array_normal))

# third
x_array_uniform = uniform_distribution()
intervals = get_interval(x_array_uniform, 10)
intervals_with_values = get_actual_interval(x_array_uniform, intervals)

print("mean = " + str(np.mean(x_array_uniform)))
print("pvariance = " + str(np.var(x_array_uniform)))
print(*intervals_with_values, sep="\n")

x_exp_array_uniform = []
for item in intervals:
    x_exp_array_uniform.append(get_exp_uniform_value(item[1][0], item[1][1], x_array_uniform))

chi_value = get_chi_values(x_exp_array_uniform, intervals_with_values, intervals)

print("Observed Chi-squared: " + str(chi_value[0]))
print("Expected Chi-squared: " + str(chi_value[1]))

draw_histogram(x_array_uniform, 0.1)
