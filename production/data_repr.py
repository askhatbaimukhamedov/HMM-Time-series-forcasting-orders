import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataRepresentation(object):
    """ Класс реализует основные
        методы отображения данных
    """
    def __init__(self):
        pass

    @staticmethod
    def set_repr_settings(seaborn_style="darkgrid", max_rows=1000, max_colums=100):
        """ Задает настройки представления данных
        """
        sns.set(style=seaborn_style)
        warnings.filterwarnings("ignore")
        pd.set_option('display.max_rows', max_rows)
        pd.set_option('display.max_columns', max_colums)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print("Настройки визуализации данных были успешно установлены!!!")

    @staticmethod
    def box_plot(data_frame, f_name):
        """ Сторит 'блоки с усами' распределений числовых предикторов.
        """
        plt.figure(figsize=(10, 4))
        plt.title('Распределение предиктора: ' + f_name)

        x_min = int(data_frame[f_name].min() - (abs(data_frame[f_name].min()) * 0.1))
        x_max = int(data_frame[f_name].max() + (abs(data_frame[f_name].max()) * 0.1))

        if x_min == 0: x_min = -1
        if x_max == 0: x_max = 1

        plt.xlim(x_min, x_max)
        sns.boxplot(x=data_frame[f_name])

    @staticmethod
    def plot_feature_importances(importances, indices, features, title, dimensions):
        """ Метод строит график feature importance для модели
        """
        plt.figure(figsize=dimensions)
        plt.title(title)
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()
