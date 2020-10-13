import numpy as np
import pandas as pd


class DataLoader(object):
    """ Далее здесь будут загружаться данные через sql...
    """
    def __init__(self):
        pass

    def __load_sql(self):
        pass

    def __update_train(self):
        pass

    @staticmethod
    def load_data(path_train, path_test, path_items, path_deliv, separator=','):
        """  Загружает необходимые датасеты
        """
        train = pd.read_csv(path_train, separator)
        test = pd.read_csv(path_test, separator)
        items = pd.read_csv(path_items, separator)
        deliv = pd.read_csv(path_deliv, separator)
        print('Все датасеты успешно загружены!!!\n')

        return train, test, items, deliv
