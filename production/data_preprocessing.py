import numpy as np
import pandas as pd
import headers as head


class DataPreprocessing(object):
    """ Класс реализует методы необходимые
        для EDA и предобработки данных
    """
    def __init__(self):
        pass

    @staticmethod
    def merge_items_charac(data_frame, df_name='items'):
        """ Объединяет предикторы item_id
            и item_charac_id в кортеж:
        """
        merge_lst = []
        pred_to_drop = ['item_id', 'item_charac_id']
        for item in zip(data_frame.item_id, data_frame.item_charac_id):
            merge_lst.append(item)
        data_frame['item_ids_char'] = merge_lst

        if df_name != 'items':
            pred_to_drop.append('real_date')
        data_frame.drop(pred_to_drop, axis=1, inplace=True)

    @staticmethod
    def add_date_block_num(data_frame):
        """ Добавляет предикторы date_block_num и real_data
        """
        lst_features = []
        data_frame['real_date'] = pd.to_datetime(data_frame.date, format='%d.%m.%Y %H:%M:%S')
        years = pd.Series(pd.to_datetime(data_frame['real_date'].unique()).year).unique()
        num_current_year = 0

        for item in data_frame.real_date:
            if item.year > years[num_current_year]:
                lst_features.append(item.isocalendar()[1] + head.SHIFT_WEEK_YEAR * (num_current_year + 1))
                num_current_year += 1
            lst_features.append(item.isocalendar()[1] + head.SHIFT_WEEK_YEAR * num_current_year)
        data_frame['date_block_num'] = pd.Series(lst_features).astype('int')

    @staticmethod
    def na_handler(data_frame, replace_type):
        """ Обрабатывает пропущенные
            значения в датасете data_frame
        """
        # Заменим пустые строки и <NULL> на np.nan
        # чтобы pandas распознал их как пропущенные
        data_frame.replace(['', '<NULL>'], np.nan, inplace=True)

        if replace_type == 'fill_zero':
            data_frame.fillna(0, inplace=True)
        elif replace_type == 'drop':
            data_frame.dropna(inplace=True)
        print(data_frame.isna().sum(), '\n')
        return data_frame

    @staticmethod
    def to_numeric_features(data_frame, name_data='train'):
        """ Меняет строковые значения предикторов на числовые:
        """
        item = []
        warehouse = []
        fabricator = []
        category = []
        charac = []

        if name_data in ['train', 'test', 'items']:
            item = [int(id_str[2:]) for id_str in data_frame.item_id]
            charac = [int(id_str) for id_str in data_frame.item_charac_id]

            if name_data in ['train', 'test']:
                warehouse = [int(id_str[2:]) for id_str in data_frame.warehouse_id]

            if name_data == 'items':
                fabricator = [int(id_str) for id_str in data_frame.fabricator_id]
                category = [int(id_str) for id_str in data_frame.item_category_id]

        if name_data == 'deliv':
            warehouse = [int(id_str[2:]) for id_str in data_frame.warehouse_id]

        return item, warehouse, fabricator, category, charac

    @staticmethod
    def compress_columns(data_frame, columns, keyword, search_type, datatype):
        """ Сжимает столбцы указанные в columns
        """
        valid_features = []

        if search_type == 'in':
            valid_features = [x for x in columns if keyword in x]
        elif search_type == 'start':
            valid_features = [x for x in columns if x.startswith(keyword)]

        if len(valid_features):
            for f in valid_features:
                data_frame[f] = data_frame[f].round().astype(datatype)
        return data_frame

    def data_compression(self, df):
        """ Делает сжатие всех предикторов в датасете df
        """
        features = df.columns.values

        # Оригинальные предикторы
        if 'date_block_num' in features:
            df['date_block_num'] = df['date_block_num'].astype(np.int16)
        if 'warehouse_id' in features:
            df['warehouse_id'] = df['warehouse_id'].astype(np.int16)
        if 'item_price' in features:
            df['item_price'] = df['item_price'].astype(np.float32)
        if 'item_category_id' in features:
            df['item_category_id'] = df['item_category_id'].astype(np.int16)
        if 'item_id_avg_item_price' in features:
            df['item_id_avg_item_price'] = df['item_id_avg_item_price'].astype(np.float32)

        # Mean encoded предикторы & lag-ые предикторы
        df = self.compress_columns(df, features, 'item_ids_char_sum_item_cnt_day', 'in', np.int32)
        df = self.compress_columns(df, features, 'item_ids_char_avg_item_cnt_day', 'in', np.float32)

        df = self.compress_columns(df, features, 'warehouse_id_avg_item_price', 'in', np.float32)
        df = self.compress_columns(df, features, 'warehouse_id_sum_item_cnt_day', 'in', np.int32)
        df = self.compress_columns(df, features, 'warehouse_id_avg_item_cnt_day', 'in', np.float32)

        df = self.compress_columns(df, features, 'item_category_id_avg_item_price', 'in', np.float32)
        df = self.compress_columns(df, features, 'item_category_id_sum_item_cnt_day', 'in', np.int32)
        df = self.compress_columns(df, features, 'item_category_id_avg_item_cnt_day', 'in', np.float32)

        df = self.compress_columns(df, features, 'item_cnt_day', 'start', np.int16)

        return df
