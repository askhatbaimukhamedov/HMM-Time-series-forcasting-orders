import gc
import time
import pickle
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class Modeling(object):
    """ Класс реализует методы машинного обучения
        для прогнозирования объема заказов товаров
    """
    def __init__(self):
        pass

    @staticmethod
    def __serialize_model(model, model_name):
        """ Сериализует модель машинного обучения
        """
        with open('%s_model.pickle' % model_name, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def __save_ans_model(model, model_name, test_x, test):
        """ Сохраняет вывод модели в одноименный файл
        """
        test_pred = model.predict(test_x)

        submission = pd.DataFrame({'ID': test.index, 'item_cnt_week': test_pred})
        submission.to_csv('%s_submission.csv' % model_name, index=False)

    @staticmethod
    def __get_predictions(model, train_x, val_x, test_x, train_y, val_y, metric):
        """ Делает прогнозы и выводит информацию о качестве модели
        """
        # Получим прогнозы
        train_pred = model.predict(train_x)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)

        predictions = {
            'train_prediction': train_pred,
            'val_prediction': val_pred,
            'test_prediction': test_pred,
        }

        # Выичлим среднеквадратичную ошибку на обучении и валидации
        if metric == 'rmse':
            predictions['train_rmse'] = mean_squared_error(train_y, train_pred, squared=False)
            predictions['val_rmse'] = mean_squared_error(val_y, val_pred, squared=False)

        # Выичлим среднюю абсолютную ошибку на обучении и валидации
        if metric == 'mae':
            predictions['train_mae'] = mean_absolute_error(train_y, train_pred)
            predictions['val_mae'] = mean_absolute_error(val_y, val_pred)

        return predictions

    def ridge_regression_model(self, train_x, train_y, val_x, val_y, test):
        """ Модель линенйной регрессии RIDGE
        """
        # Нормировка датасета train_x
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_x.values)
        train_x_norm = scaler.transform(train_x.values)
        val_x_norm = scaler.transform(val_x.values)
        test_norm = scaler.transform(test.values)

        gc.collect()
        ts = time.time()

        lm = linear_model.Ridge()
        lm.fit(train_x_norm, train_y)
        print('Время обучения: %s сек.' % (time.time() - ts))

        # Качество работы модели и тестовый прогноз
        return self.__get_predictions(lm, train_x_norm, val_x_norm, test_norm,
                                                 train_y, val_y, metric='rmse')

    def xgboost_model(self, train_x, train_y, val_x, val_y, test):
        """ Модель XGBoost
        """
        gc.collect()
        ts = time.time()

        xgb_train = xgb.DMatrix(train_x.values, train_y.values)
        param = {'max_depth': 9,            # Maximum depth of a tree
                 'subsample': 1,            # Subsample ratio of the training instances
                 'min_child_weight': 0.5,   # Minimum sum of instance weight (hessian) needed in a child
                 'eta': 0.3,                # Step size shrinkage used in update to prevents overfitting
                 'lambda': 5,               # L2 regularization term on weights
                 'num_round': 1000,         # The number of rounds for boosting
                 'seed': 1,                 # Random number seed
                 'verbosity': 2,            # Verbosity of printing messages
                 'eval_metric': 'rmse'}     # Mean absolute error or root mean squared error

        model_xgb = xgb.train(param, xgb_train)
        print('Время обучения: %s мин.' % ((time.time() - ts) // 60))

        # Качество работы модели и тестовый прогноз
        return self.__get_predictions(model_xgb, xgb.DMatrix(train_x.values), xgb.DMatrix(val_x.values),
                                                                              xgb.DMatrix(test.values),
                                                                              train_y, val_y, metric='rmse')

    def random_forest_model(self, train_x, train_y, val_x, val_y, test):
        """ Модель случайного леса
        """
        gc.collect()
        ts = time.time()

        rand_forest = RandomForestRegressor(
            bootstrap=True,
            max_depth=30,
            max_features=3,
            min_samples_leaf=5,
            min_samples_split=12,
            n_estimators=250,
            random_state=42,
            verbose=1,
            n_jobs=-1
        )

        rand_forest.fit(train_x.values, train_y.values)
        print('Время обучения: %s мин.' % ((time.time() - ts) / 60))

        # Качество работы модели и тестовое предсказание
        return self.__get_predictions(rand_forest, train_x, val_x,
                                      test, train_y, val_y, metric='rmse')
