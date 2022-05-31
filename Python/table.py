from typing import Dict, Tuple
import pandas as pd
from matplotlib import pyplot as plt

from variance_collection import VarianceCollection


class Table:
    def __init__(self, x: VarianceCollection, y: VarianceCollection, frequency_table: Dict[Tuple[int, int], int]):
        self.x_intervals = x
        self.y_intervals = y
        self.table = frequency_table

    def get_frequency_at(self, x_index, y_index):
        return self.table.get((x_index, y_index), 0)

    def markdownTable(self, data, columns, rows):
        """
        > The function takes a list of lists, and returns a list of strings, where each string is a row of a markdown table

        :param data: a list of lists of data
        :param columns: A list of column names
        :param rows: The list of rows to be displayed in the table
        :return: A list of strings.
        """
        return  [
            '|'+'|'.join(map(lambda x: str(x), g)) + '|' for g in
            [["Y\\X"] + columns] + [[data[i][j] for j in range(len(columns))] for i in range(len(rows))]
        ]

    def draw(self):
        columns = [f"[{x[0]}, {x[1] + 1})" for x in self.x_intervals.segments()]

        rows = [f"[{y[0]}, {y[1] + 1})" for y in self.y_intervals.segments()]

        data = []
        for y in range(len(rows)):
            data.append([])
            data[y].append(rows[y])
            for x in range(len(columns)):
                data[y].append(self.table.get((x, y), ""))

        fig, ax = plt.subplots()

        fig.patch.set_visible(False)
        ax.axis('off')

        df = pd.DataFrame(data, columns=["Y\\X"] + columns)

        ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc="center")

        fig.tight_layout()

        plt.savefig("results/table.png")
        table_markdown = self.markdownTable(data, columns, rows)

        for i in table_markdown:
            print(i)
        plt.show()


