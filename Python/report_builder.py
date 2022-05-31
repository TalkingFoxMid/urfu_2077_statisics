import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import laplace

from statistics_calculator import StatisticsCalculator
from statistics_data import StatisticsData
from table_helper import draw_table
from variance_collection import VarianceCollection

class NormReportOptions:
    def __init__(self, alpha: float, xi_expected: float | None = None, q: float | None = None):
        self.xi_expected = xi_expected
        self.q = q
        self.alpha = alpha


def calculate_laplace(a, sigma, left, right):
    """
    It calculates the probability that a Laplace random variable with mean $a$ and scale parameter $\sigma$ is between
    $left$ and $right$

    :param a: the mean of the distribution
    :param sigma: the standard deviation of the Laplace distribution
    :param left: the left bound of the interval
    :param right: the right bound of the interval
    :return: The probability that a random variable X is between left and right.
    """
    return laplace.cdf((right - a) / sigma) - laplace.cdf((left - a) / sigma)


class ReportBuilder:
    def __init__(self, intervals: VarianceCollection, stat: StatisticsData, name,
                 options: NormReportOptions):
        self.name = name
        self.collection = intervals
        self.options = options
        self.stat = stat

    def print_table(self):
        """
        It takes a list of intervals and a name, and then it prints a table with the name of the distribution as the first
        column, and the number of values in each interval as the other columns
        """
        values = {x.middle: x.count for x in self.collection.intervals}
        omegas = {x.middle: "${0} / {1}$".format(f"{{{x.count}}}", f"{{{sum(map(lambda x: x.count, self.collection.intervals))}}}") for x in self.collection.intervals}
        columns = [self.name] + list(map(str, values.keys()))
        data = [[f"$n_{{{self.name}_i}}$"] + list(map(str, values.values())),
                ["ω"] + list(map(str, omegas.values()))]
        draw_table(columns, data, filename=f"results/Distribution {self.name}.png")

    def create_hist_and_polygones(self):
        """
        It creates a histogram and a polygon of the frequency distribution of the given variable
        """
        frequencies = {f"{(x.left+x.right+1)/2}": x.count for x in self.collection.intervals}
        values = list(frequencies.values())
        indexes = list(map(lambda x: str(x).replace(".0", ""), frequencies.keys()))
        print(self.collection.intervals)

        df = pd.DataFrame(dict(data=values), index=indexes)
        fig, ax = plt.subplots()
        df["data"].plot(kind='bar', edgecolor='black', grid=True, width=1.0)
        df["data"].plot(kind='line', marker='*', color='black', ms=10, grid=True)
        plt.title(f"Гистограмма и полигоны варианты {self.name}")
        plt.xlabel(self.name)
        plt.ylabel("n")

        step = 10
        ticks = list(range(step, max(values) + step, step)) + values
        major_tick = list(sorted(ticks))
        ax.set_yticks(major_tick)

        plt.savefig(f"results/Hist_Polygones {self.name}.png")
        plt.show()

    def create_polygones(self):
        frequencies = {f"{(x.left+x.right+1)/2}": x.count for x in self.collection.intervals}
        values = list(frequencies.values())
        indexes = list(map(lambda x: str(x).replace(".0", ""), frequencies.keys()))
        print(self.collection.intervals)

        df = pd.DataFrame(dict(data=values), index=indexes)
        fig, ax = plt.subplots()
        df["data"].plot(kind='line', marker='*', color='black', ms=10, grid=True)
        plt.title(f"Полигон частот признака {self.name}")
        plt.xlabel(self.name)
        plt.ylabel("n")

        step = 10
        ticks = list(range(step, max(values) + step, step)) + values
        major_tick = list(sorted(ticks))
        ax.set_yticks(major_tick)

        plt.savefig(f"results/Polygones {self.name}.png")
        plt.show()

    def create_hist(self):
        frequencies = {f"{x.left}-{x.right + 1}": x.count for x in self.collection.intervals}
        values = list(frequencies.values())
        indexes = list(map(lambda x: str(x).replace(".0", ""), frequencies.keys()))
        print(self.collection.intervals)

        df = pd.DataFrame(dict(data=values), index=indexes)
        fig, ax = plt.subplots()
        df["data"].plot(kind='bar', edgecolor='black', grid=True, width=1.0)
        plt.title(f"Гистограмма относительных частот признака {self.name}")
        plt.xlabel(self.name)
        plt.ylabel("n")

        step = 10
        ticks = list(range(step, max(values) + step, step)) + values
        major_tick = list(sorted(ticks))
        ax.set_yticks(major_tick)

        plt.savefig(f"results/Hist {self.name}.png")
        plt.show()

    def create_norm_hist(self, a, sigma):
        frequencies = {x.middle: x.count for x in self.collection.intervals}
        values = list(frequencies.values())
        indexes = list(frequencies.keys())

        n = self.collection.count()
        good_values = [calculate_laplace(a, sigma, x.left, x.right) * n for x in self.collection.intervals]

        plt.figure(figsize=(10, 10))
        plt.plot(indexes, values, label=r'Эмпирические данные')
        plt.plot(indexes, good_values, label=fr'Ожидаемые данные')
        plt.xlabel(r'$x$', fontsize=7)
        plt.ylabel(r'$f(x)$', fontsize=7)
        plt.grid(True)
        plt.legend(loc='best', fontsize=7)

        plt.savefig(f"results/Hist_Norm_{self.name}.png")
        plt.show()

    def print_stat(self):
        stat = self.stat
        columns = ["Числовые хар-ки", self.name]
        data = [
            ["Выборочное среднее", stat.sample_mean],
            ["Выборочная дисперсия", stat.sample_variance],
            ["Выборочное СКО", stat.sigma],
            ["\"Испавленное\" СКО", stat.sigma_2],
            ["Мода", stat.mode],
            ["Медиана", stat.median],
        ]

        draw_table(columns, data, filename=f"results/Characteristics {self.name}.png")

    def report_norm(self):
        """
        It creates a table with the point and interval estimates of the mean and variance, and the p-value of the Pearson
        test
        """
        t_gamma = 1 - self.options.alpha
        stat = self.stat
        n = self.collection.count()
        k = t_gamma * stat.sigma_2 / (n ** 0.5)
        a = stat.sample_mean
        sigma = stat.sigma_2
        columns = ["Числовые хар-ки", self.name]
        if self.options.q is None:
            q = float(input(f"Значение в таблице q=q({t_gamma}, {n}): "))
        else:
            q = self.options.q
        p = self.pirson(a, sigma)
        data = [
            [r"Точечная оценка $a$", a],
            [r"Точечнкая оценка $\sigma$", sigma],
            [r"Интервальная оценка $a$", f"({stat.sample_mean - k}; {stat.sample_mean + k})"],
            [r"Интервальная оценка $\sigma$", f"({stat.sigma_2 * (1 - q)}; {stat.sigma_2 * (1 + q)})"],
            [r"Критерий Пирсона", p]
        ]
        draw_table(columns, data, filename=f"results/Characteristics Normal {self.name}.png")
        self.create_norm_hist(a, sigma)

    def pirson(self, a, sigma):
        """
        It checks if the empirical value of the Pirson criterion is less than the expected value

        :param a: the average value of the sample
        :param sigma: the standard deviation of the distribution
        :return: True if the empirical value of the Pirson criterion is less than the expected value.
        """
        n = self.collection.count()
        s = 0
        for interval in self.collection.intervals:
            p1_i = interval.count
            p2_i = calculate_laplace(a, sigma, interval.left, interval.right) * n
            s += (p2_i - p1_i) ** 2 / p2_i
        xi_empirical = s
        k = len([v for v in self.collection.intervals if v.count >= 4]) - 3
        if self.options.xi_expected is not None:
            xi_expected = self.options.xi_expected
        else:
            xi_expected = float(input(f"Значение в таблице Пирсона Xi = Xi({self.options.alpha}, {k}): "))
        return xi_empirical < xi_expected
