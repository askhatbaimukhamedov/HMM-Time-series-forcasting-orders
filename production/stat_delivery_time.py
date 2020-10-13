import numpy as np
import pandas as pd


class StatDeliveryTime(object):
    """ Класс реализует основные методы для расчета статистики
        по срокам доставок от производителей до складов компании
    """

    def __init__(self):
        pass

    @staticmethod
    def __percentile(num_percent):
        def percentile_(delivery_time):
            return np.percentile(delivery_time, num_percent)

        percentile_.__name__ = 'percentile_%s' % num_percent
        return percentile_

    def tt(self, d_time):
        # Из типа объект делаем датыот
        d_time['ДатаПоступленияНаСклад'] = pd.to_datetime(d_time.ДатаПоступленияНаСклад, format='%d.%m.%Y %H:%M:%S')
        d_time['ДокументРезерваДатаСоздания'] = pd.to_datetime(d_time.ДокументРезерваДатаСоздания,
                                                               format='%d.%m.%Y %H:%M:%S')

        # Создадим предиктор со сроками поставок
        d_time['difference'] = d_time.ДатаПоступленияНаСклад - d_time.ДокументРезерваДатаСоздания

        # Выберем только количество
        # дней в сроках поставок
        lst = []
        for item in d_time.difference:
            lst.append(item.days)
        d_time['delivery'] = lst

