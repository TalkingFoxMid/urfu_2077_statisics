import math
from collections import Counter

from matplotlib import pyplot as plt

from correlation_options import CorrelationOptions
from statistics_data import StatisticsData
from table import Table
from variance_collection import VarianceCollection



def get_middle(interval: VarianceCollection, index: int):
    """
    It returns the middle of the interval at the given index

    :param interval: VarianceCollection
    :type interval: VarianceCollection
    :param index: the index of the interval you want to get the middle of
    :type index: int
    :return: The middle of the interval
    """
    return interval.get_interval(index).middle


def get_conditional_coords(pairs):
    """
    It takes a list of pairs, and returns a list of pairs, where the first element of each pair is the same as the first
    element of the input pairs, and the second element of each pair is the average of the second elements of the input pairs
    that have the same first element

    :param pairs: a list of (x, y) pairs
    :return: A list of tuples, where each tuple is a pair of a mean and a conditional mean.
    """
    result = {}
    for x, y in pairs:
        if x not in result:
            result[x] = []
        result[x].append(y)
    result = [(xm, Counter(yms)) for xm, yms in result.items()]
    result = [(xm, sum(y * n for y, n in c.items()) / sum(c.values())) for xm, c in result]
    return list(sorted(result, key=lambda t: t[0]))


class Correlation:
    def __init__(self, table: Table, x_stat: StatisticsData, y_stat: StatisticsData, options: CorrelationOptions):
        self.table = table
        self.x_stat = x_stat
        self.y_stat = y_stat
        self.count = sum(self.table.table.values())
        self.options = options

    def coefficient(self):
        """
        We calculate the covariance of the two variables by multiplying the frequency of each pair of values by the middle
        of the interval of the x-value and the middle of the interval of the y-value, and then summing all of these products
        :return: The correlation coefficient.
        """
        frequencies = self.table.table
        x_intervals = self.table.x_intervals
        y_intervals = self.table.y_intervals
        n = self.count

        s = sum(v * get_middle(x_intervals, x) * get_middle(y_intervals, y) for (x, y), v in frequencies.items())
        kxy = (s - n * self.x_stat.sample_mean * self.y_stat.sample_mean) / (n - 1)
        return kxy / (self.x_stat.sigma_2 * self.y_stat.sigma_2)

    def significance(self):
        """
        The function returns True if the absolute value of the t-statistic is less than the critical value of t
        :return: the value of the significance of the correlation coefficient.
        """
        n = self.count
        rxy = self.coefficient()
        t = rxy * math.sqrt((n - 2) / (1 - rxy ** 2))
        if self.options.critical_t is None:
            critical_t = float(input(f"Значение в таблице Стьюдента t=t({self.options.alpha}, {n - 2}): "))
        else:
            critical_t = self.options.critical_t
        return abs(t) < critical_t

    def regression_lines(self):
        r = self.coefficient()
        y0 = self.y_stat.sample_mean
        x0 = self.x_stat.sample_mean

        def y_regression(x):
            k = r * self.y_stat.sigma_2 / self.x_stat.sigma_2
            return y0 + k * (x - x0)

        def x_regression(y):
            k = r * self.x_stat.sigma_2 / self.y_stat.sigma_2
            return x0 + k * (y - y0)

        # Y = Y(X)
        xs, ys = list(map(list, zip(*self.get_x_to_y())))

        plt.figure(figsize=(10, 10))
        plt.plot(xs, ys, label=r'Эмпирические данные')
        plt.plot(xs, list(map(y_regression, xs)), label=r'Теоретические данные')
        plt.grid(True)
        plt.legend(loc='best', fontsize=7)

        plt.savefig(f"results/Regression X to Y.png")
        plt.show()

        # X = X(Y)
        xs, ys = list(map(list, zip(*self.get_y_to_x())))

        plt.figure(figsize=(10, 10))
        plt.plot(xs, ys, label=r'Эмпирические данные')
        plt.plot(xs, list(map(x_regression, xs)), label=r'Теоретические данные')
        plt.grid(True)
        plt.legend(loc='best', fontsize=7)

        plt.savefig(f"results/Regression Y to X.png")
        plt.show()

    def get_x_to_y(self):
        """
        It takes a list of coordinates and returns a list of coordinates that are in the same order as the input list, but
        with the x and y coordinates swapped
        :return: The conditional coords of the middles of the squares.
        """
        return get_conditional_coords([(x, y) for x, y in self.get_middles()])

    def get_y_to_x(self):
        return get_conditional_coords([(y, x) for x, y in self.get_middles()])

    def get_middles(self):
        for x, y in self.table.table.keys():
            xm = get_middle(self.table.x_intervals, x)
            ym = get_middle(self.table.y_intervals, y)
            yield xm, ym
