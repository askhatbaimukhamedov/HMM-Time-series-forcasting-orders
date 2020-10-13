import gc
import numpy as np
import pandas as pd
from itertools import product
import time

import headers as head
import data_loader as loader
import data_repr as represent
import data_preprocessing as preproc
import modeling as mod


class PreProcessingError(Exception):
    """ Ошибки возникающие при
        предобработке данных
    """
    def __init__(self, msg):
        self.message = msg


class Pipeline(object):
    """ Конвейр построения модели
        прогнозирующей объем заказов
    """
    def __init__(self):
        self.preprocess = preproc.DataPreprocessing()
        self.model = mod.Modeling()

    def __create_grid(self, data_frame):
        """ Для каждой недели создает таблицу всевозможных пар склад/товар
        """
        index_cols = ['warehouse_id', 'item_ids_char', 'date_block_num']
        grid = []

        for block_num in data_frame['date_block_num'].unique():
            cur_warehouses = data_frame.loc[data_frame['date_block_num'] == block_num, 'warehouse_id'].unique()
            cur_items = data_frame.loc[data_frame['date_block_num'] == block_num, 'item_ids_char'].unique()
            grid.append(np.array(list(product(*[cur_warehouses, cur_items, [block_num]])), dtype='object'))

        # Соберем таблицу и сделаем сжатие данных
        grid = pd.DataFrame(np.vstack(grid), columns=index_cols)
        return self.preprocess.data_compression(grid)

    def __group_by_item_warehouse(self, items, data_frame, grid):
        """ Группирует датасет train по неделям, складу,
            товару. Агрегирует по сумме заказов, средней цене
        """
        items_tmp = items.drop('item_price', axis=1)

        # Группировка товаров по неделям, региону, товару
        data_frame_m = data_frame.groupby(['date_block_num', 'warehouse_id', 'item_ids_char']). \
            agg({'item_cnt_day': 'sum', 'item_price': np.mean}).reset_index()
        data_frame_m = pd.merge(grid, data_frame_m, on=['date_block_num',
                                                        'warehouse_id',
                                                        'item_ids_char'], how='left').fillna(0)
        data_frame_m = pd.merge(data_frame_m, items_tmp, on='item_ids_char', how='left')
        data_frame_m = self.preprocess.na_handler(data_frame_m, replace_type='fill_zero')
        return self.preprocess.data_compression(data_frame_m)

    def __create_mean_encodings(self, data_frame, data_frame_m):
        """ Метод создаёт mean encoded предикторы
        """
        for type_id in ['item_ids_char', 'warehouse_id', 'item_category_id']:
            for column_id, aggregator, aggtype in [('item_price', np.mean, 'avg'),
                                                   ('item_cnt_day', np.sum, 'sum'),
                                                   ('item_cnt_day', np.mean, 'avg')]:
                mean_df = data_frame.groupby([type_id, 'date_block_num']).aggregate(aggregator). \
                                              reset_index()[[column_id, type_id, 'date_block_num']]
                mean_df.columns = [type_id + '_' + aggtype + '_' + column_id, type_id, 'date_block_num']
                data_frame_m = pd.merge(data_frame_m, mean_df, on=['date_block_num', type_id], how='left')
                del mean_df
                gc.collect()

        return self.__fill_mean_zero_encode(data_frame_m)

    def __create_lag_features(self, data_frame_m, train_m):
        """ Метод создает лаговые предикторы
        """
        lag_features = list(data_frame_m.columns[8:]) + ['item_cnt_day']

        for lag in head.LAGS:
            train_new_df = train_m.copy()
            train_new_df['date_block_num'] += lag
            train_new_df = train_new_df[['date_block_num', 'warehouse_id', 'item_ids_char'] + lag_features]
            train_new_df.columns = ['date_block_num',
                                    'warehouse_id',
                                    'item_ids_char'] + [x + '_lag_' + str(lag) for x in lag_features]
            data_frame_m = pd.merge(data_frame_m, train_new_df, on=['date_block_num', 'warehouse_id',
                                                                    'item_ids_char'], how='left')
            del train_new_df
            gc.collect()
            print('lag %s успешно создан' % lag)
        return self.__fill_mean_zero_encode(data_frame_m)

    def __fill_mean_zero_encode(self, data_frame):
        """ Метод заполняет значения mean encodings
            и лаговые предикторы (0-ем или средним)
        """
        f = ['item_cnt', 'item_category_id']
        for feature in data_frame.columns:
            if any(st in feature for st in f):
                data_frame[feature] = data_frame[feature].fillna(0)
            elif 'item_price' in feature:
                data_frame[feature] = data_frame[feature].fillna(data_frame[feature].median())
        return self.preprocess.data_compression(data_frame)

    def __to_numeric_dataset(self, data_frame, name='items'):
        """ Метод заменяет строковые предикторы на числовые
        """
        item, warehouse, fabricator, category, charac = \
            self.preprocess.to_numeric_features(data_frame, name_data=name)

        if name in ['train', 'test', 'items']:
            data_frame['item_id'] = item
            data_frame['item_charac_id'] = charac
            if name in ['train', 'test']:
                data_frame['warehouse_id'] = warehouse
            if name == 'items':
                data_frame['fabricator_id'] = fabricator
                data_frame['item_category_id'] = category
        elif name == 'deliv':
            data_frame['warehouse_id'] = warehouse

    @staticmethod
    def __cols_to_drop(data_frame, train_m, lag_features):
        print('Columns to drop\n', lag_features[:-1] + ['item_price'])

        train_cols = train_m.columns.values
        test_cols = data_frame.columns.values

        for col_to_drop in lag_features[:-1] + ['item_price']:
            if col_to_drop in train_cols:
                train_m = train_m.drop(col_to_drop, axis=1)
            if col_to_drop in test_cols:
                data_frame = data_frame.drop(col_to_drop, axis=1)

    @staticmethod
    def delete_discharge(df):
        """ Метод удаляет выбросы и отрицательные значения:
            item_cnt_day < (99 percentile * 3); item_price < 95 percentile
        """
        df = df[(df['item_cnt_day'] < df.describe([.95, .99]).item_cnt_day['99%'] * 3) & (df['item_cnt_day'] >= 0)]
        df = df[(df['item_price'] < df.describe([.95, .99]).item_price['95%']) & (df['item_price'] >= 0)]
        print('Выбросы были успешно удалены!!!\n')
        return df

    def prepare_data(self, *args):
        """ Добавляет предикторы date_block_num, ral_date
            и обрабатывает пропущенные значения"""
        if len(args):
            self.preprocess.add_date_block_num(args[0])
            for i in range(len(args)):
                if i <= 1:
                    self.preprocess.na_handler(args[i], replace_type='fill_zero')
                else:
                    self.preprocess.na_handler(args[i], replace_type='drop')
            print('Пропущенные значения были успешно обработаны!!!\n')
        else:
            raise PreProcessingError('Пустой список датасетов')

    def transform_data_for_predict(self, data_frame, items, deliv):
        """ Преобразовывает датасет в подходящий
            вид для обучения моделей
        """
        # Создадим таблицу товары/склады/недели
        grid = self.__create_grid(data_frame)

        # Сгруппируем товары по неделям, по региону, по товару и добавим предиктор срок доставки
        data_frame_m = self.__group_by_item_warehouse(items, data_frame, grid)
        data_frame_m = pd.merge(data_frame_m, deliv, on=['warehouse_id', 'fabricator_id'], how='left'). \
            fillna(np.mean(deliv.delivery))
        print('Grid была успешно создана!!!')
        return data_frame_m

    def merge_datasets(self, train, items, deliv):
        """ Объединяет датасеты полученные в параметрах:
            train & items и записывает результат в train
        """
        # Категориальные предикторы --> числовые
        self.__to_numeric_dataset(train, name='train')
        self.__to_numeric_dataset(items, name='items')
        self.__to_numeric_dataset(deliv, name='deliv')

        self.preprocess.merge_items_charac(items)
        self.preprocess.merge_items_charac(train, df_name='train')

        print('Датасеты были успешно объеденины!!!')
        return self.preprocess.na_handler(pd.merge(train, items, how='left', on='item_ids_char'),
                                                                       replace_type='fill_zero')

    def make_lag_mean_encode_features(self, data_frame, data_frame_m):
        """ Добавляет mean encodings & lag-ые предикторы
        """
        # Создадим mean encodings предикторы
        data_frame_m = self.__create_mean_encodings(data_frame, data_frame_m)
        print('Mean Encodings предикторы были успешно созданы!!!')

        # Создадим лаговые предикторы
        return self.__create_lag_features(data_frame_m, data_frame_m)

    @staticmethod
    def split_data(df):
        # Данные слишком далекого прошлого считаются менее релевантными
        df = df[df['date_block_num'] > head.RELEVANT_WEEK]

        # Разделение датасета на обучение и валидацию
        train_set = df[df['date_block_num'] <= head.SUP_INDEX_TRAIN]
        val_set = df[(df['date_block_num'] > head.SUP_INDEX_TRAIN) & (df['date_block_num'] < head.SUP_INDEX_VAL)]

        # Предиктор item_ids_char разделим на: item_id, item_charc_id и удалим его
        val_set[['item_id', 'item_charac_id']] = pd.DataFrame(val_set['item_ids_char'].tolist(), index=val_set.index)
        train_set[['item_id', 'item_charac_id']] = pd.DataFrame(train_set['item_ids_char'].tolist(),
                                                                             index=train_set.index)
        train_set.drop('item_ids_char', axis=1, inplace=True)
        val_set.drop('item_ids_char', axis=1, inplace=True)

        # Разделим датасеты на предикторы и таргет
        train_x = train_set.drop(['item_cnt_day'], axis=1)
        train_y = train_set['item_cnt_day']
        val_x = val_set.drop(['item_cnt_day'], axis=1)
        val_y = val_set['item_cnt_day']
        features = list(train_x.columns.values)
        del df, train_set, val_set
        gc.collect()
        return train_x, train_y, val_x, val_y

    def handle_test_data(self, train_m, data_frame, items, deliv):
        """ Обрабатывает тестовый набор
        """
        # Сделаем числовые предикторы
        self.__to_numeric_dataset(data_frame, name='test')
        # Объеденим предикторы item_id & item_charac_id
        self.preprocess.merge_items_charac(data_frame, df_name='test')
        # Добавим предиктор item_price
        data_frame = pd.merge(data_frame, items, how='left', on='item_ids_char')
        # Обработаем пропуски после объединения
        data_frame = self.preprocess.na_handler(data_frame, replace_type='fill_zero')

        data_frame = self.transform_data_for_predict(data_frame, items, deliv)
        # Добавим к комбинации товар-склад предиктор срок доставки
        data_frame = pd.merge(data_frame, deliv, on=['warehouse_id', 'fabricator_id'], how='left'). \
            fillna(np.mean(deliv.delivery))
        # Удалим наблюдения из датасета где встречаются пропущенные значения
        data_frame['fabricator_id'] = data_frame['fabricator_id'].fillna(0)
        data_frame.to_csv('orders_test.csv', index=False)
        data_frame.drop('item_cnt_day', axis=1, inplace=True)

        data_frame = self.__create_lag_features(data_frame, train_m)

        return self.__fill_mean_zero_encode(data_frame)


def main():
    ts = time.time()
    pipeline = Pipeline()
    data_loader = loader.DataLoader()
    data_repr = represent.DataRepresentation()
    model = mod.Modeling()

    # Зададим настройки отображения данных
    data_repr.set_repr_settings()

    # Загружаем ежедневные исторические данные(train/test) + товрары + сроки поставок
    train, test, items, deliv = data_loader.load_data(path_train='../datasets/train_orders.csv',
                                                      path_test='../datasets/test_orders.csv',
                                                      path_items='../datasets/items.csv',
                                                      path_deliv='../datasets/delivery_time.csv',
                                                      separator=',')
    # Дропним предикторы неважные для прогнозирования
    train.drop(['item_name', 'item_charac_name', 'warehouse_name'], axis=1, inplace=True)
    items.drop(['item_name', 'item_charac_name'], axis=1, inplace=True)

    # Подготовим данные обучения для построения модели
    pipeline.prepare_data(*[train, items, deliv])
    train = pipeline.merge_datasets(train, items, deliv)
    train = pipeline.delete_discharge(train)
    train_m = pipeline.make_lag_mean_encode_features(train, pipeline.transform_data_for_predict(train, items, deliv))

    # Преобразуем тестовый датасет
    # test.drop(['item_name', 'item_charac_name', 'warehouse_name'], axis=1, inplace=True)
    # pipeline.prepare_data(test)
    # test = pipeline.handle_test_data(train_m, test, items, deliv)

    # Обучение модели и прогнозы
    train_x, train_y, val_x, val_y = pipeline.split_data(train_m)
    output_regression = model.ridge_regression_model(train_x, train_y, val_x, val_y, test)
    output_xgboost = model.xgboost_model(train_x, train_y, val_x, val_y, test)
    output_random_forest = model.random_forest_model(train_x, train_y, val_x, val_y, test)
    print(output_regression, output_xgboost, output_random_forest)
    print('Время работы приложения: %s мин.' % ((time.time() - ts) / 60))


if __name__ == '__main__':
    main()
