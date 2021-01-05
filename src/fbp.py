import itertools
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ProphetModel():
    def __init__(self, data, area,
                 compart, group_column, date_column,
                 prediction_size=14, query='20200701 > Date',
                 holidays=True,
                 growth='linear',
                 cap=None):

        self.data = data
        self.area = area
        self.compart = compart
        self.group_column = group_column
        self.date_column = date_column
        self.query = query
        self.prediction_size = prediction_size
        self.holidays = holidays
        self.growth = growth
        self.cap = cap

    @property
    def df(self):
        return self.make_df()

    @property
    def train(self):
        return self.df[:-self.prediction_size]

    @property
    def y_true(self):
        return self.df['y'].values

    @property
    def y_pred(self):
        return self.forecast['yhat'].values

    @property
    def future_df(self):
        fdf = self.m.make_future_dataframe(periods=self.prediction_size)
        if self.growth is 'logistic':
            if self.cap is None:
                self.cap = self.df['y'].max() + 20000
            fdf['cap'] = self.cap
        return fdf

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def rmse(self):
        return mean_squared_error(self.y_true, self.y_pred, squared=False)

    def print_metrics(self):
        print('MAE: %.3f' % self.mae())
        print('MSE: %.3f' % self.mse())
        print('RMSE: %.3f' % self.rmse())

    def fit(self, **kwargs):
        self.make_df()

        self.m = Prophet(growth=self.growth, **kwargs)

        if self.holidays:
            self.m.add_country_holidays(country_name='IT')

        _ = self.m.fit(self.train)

        self.forecast = self.m.predict(self.future_df)

    def make_df(self):
        df = self.data.loc[
            (self.data[self.group_column] == self.area), :].query(self.query)

        df = df.loc[:, [self.date_column, self.compart]]
        df.columns = ['ds', 'y']

        if self.growth is 'logistic':
            if self.cap is None:
                self.cap = df['y'].max() + 20000
            df['cap'] = self.cap

        return df.reset_index(drop=True)

    def fit_cv(self, param_grid=None, initial='80 days',
               horizon='14 days', period='14 days',
               rolling_window=1):

        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            }

        all_params = [dict(zip(param_grid.keys(), v))
                      for v in itertools.product(*param_grid.values())]

        # We manually set this cap otherwise are not able
        # to see the plots clearly. If we were to set the correct cap,
        # we would have used the total pop:
        # self.df['cap'] = get_region_pop(province, pop_prov_df, prov_list_df)
        if self.growth is 'logistic':
            if self.cap is None:
                self.cap = self.df['y'].max() + 20000
            self.df['cap'] = self.cap

        rmses = []
        for params in all_params:
            m = Prophet(self.growth, **params)

            if self.holidays:
                m.add_country_holidays(country_name='IT')

            m.fit(self.df)

            df_cv = cross_validation(m, initial=initial, horizon=horizon,
                                     period=period, parallel="processes")

            df_p = performance_metrics(df_cv, rolling_window=rolling_window)

            rmses.append(df_p['rmse'].values[0])

        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses

        self.tuning_results = tuning_results
        self.best_params = all_params[np.argmin(rmses)]

        self.fit(**self.best_params)

    def plot_data(self, figsize=(8, 5)):
        plt.figure(figsize=figsize)
        plt.plot(self.df['ds'], self.df['y'])
        plt.xlabel('Date')
        plt.ylabel('Individuals')
        plt.title(self.compart)
        plt.show()

    def plot_comp(self):
        self.m.plot_components(self.forecast)
        plt.show()

    def plot(self, figsize=(8, 5)):
        plt.figure(figsize=figsize)

        plt.plot(self.df['ds'], self.y_true, label='Actual')
        plt.plot(self.df['ds'], self.y_pred, label='Predicted')

        plt.fill_between(self.df[-self.prediction_size:]['ds'],
                         self.forecast.yhat_lower[-self.prediction_size:],
                         self.forecast.yhat_upper[-self.prediction_size:],
                         alpha=.15)

        plt.axvline(self.train.iloc[-1]['ds'],
                    linestyle='dashed', color='grey', alpha=0.3)

        if self.cap is not None:
            plt.axhline(self.cap, linestyle='dashed', color='b')

        plt.legend()
        plt.show()
