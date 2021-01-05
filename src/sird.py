import pandas as pd
import numpy as np

from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def logistic_R0(t, R_0_start, k, x0, R_0_end):
    """
    R0 moduled as logistic function
    """

    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end


def beta(t, R_0_start, k, x0, R_0_end, gamma):
    """
    Computes beta at a given time `t`
    """

    return logistic_R0(t, R_0_start, k, x0, R_0_end) * gamma


def sird_calc(y, t, N, gamma, alpha, R_0_start, k, x0, R_0_end, beta):
    """
    Computes SIRD model
    """

    S, I, R, D = y
    dSdt = -beta(t, R_0_start, k, x0, R_0_end, gamma) * S * I / N
    dIdt = -dSdt - (1 - alpha) * gamma * I - alpha * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * I
    return dSdt, dIdt, dRdt, dDdt


def sird(province,
         pop_prov_df,
         gamma=1/7,
         alpha=0.01,
         days=101,
         R_0_start=2,
         k=0.2,
         x0=40,
         R_0_end=0.3,
         prov_list_df=None):
    """
    Create and compute a SIRD model

    Parameters
    ----------

    province : str
        The province name.

    pop_prov_df : pandas DataFrame
        The DataFrame with demographic data.

    gamma : float (default=1/7)
        Inverse of how many days the infection lasts.

    alpha : float (default=0.01)
        Death rate.

    days : int (default=101)
        Total number of days to predict + 1.

    R_0_start : float (default=2)
        Starting value of RO

    k : float (default=0.2)
        How quickly R0 declines. Lower values of k will
        let R0 need more time to become lower.

    x0 : int (default=40)
        Value on the x-axis of the inflection point of R0.
        This can be interpreted as the day in which lockdown
        comes into effect.

    R_0_end : float (default=0.3)
        Final value of RO

    Returns
    -------
    A numpy array of shape (4, days).
    """

    # Population
    if prov_list_df is not None:
        prov_list = prov_list_df[
            prov_list_df.Region == province
        ]['Province'].values

        N = 0
        for prov in prov_list:
            N += pop_prov_df.loc[
                (pop_prov_df.Territorio == prov) &
                (pop_prov_df.Eta == "Total")
                ]['Value'].values[0]
    else:
        N = pop_prov_df.loc[
            (pop_prov_df.Territorio == province) &
            (pop_prov_df.Eta == "Total")
            ]['Value'].values[0]

    times = range(days)

    # S0, I0, R0, D0: initial conditions vector
    init = N-1, 1, 0, 0

    # Solve the model
    sirsol = odeint(
        sird_calc,
        init,
        times,
        args=(
            N, gamma, alpha, R_0_start, k, x0, R_0_end, beta
        )
    )

    return sirsol.T


def Model(days, N, R_0_start, k, x0, R_0_end, alpha, gamma):
    y0 = N-1.0, 1.0, 0.0, 0.0,
    times = range(0, days)

    sirsol = odeint(
        sird_calc,
        y0,
        times,
        args=(
            N, gamma, alpha, R_0_start, k, x0, R_0_end, beta
        )
    )

    S, I, R, D = sirsol.T
    R0_over_time = [beta(i, R_0_start, k, x0, R_0_end, gamma)/gamma
                    for i in range(len(times))]

    return times, S, I, R, D, R0_over_time


class DeterministicSird():
    def __init__(self, data_df, pop_prov_df,
                 prov_list_df, area, group_column,
                 data_column, data_filter, lag,
                 days_to_predict, is_regional=True, pcm_data=None):

        self.data_df = data_df
        self.pop_prov_df = pop_prov_df
        self.prov_list_df = prov_list_df
        self.area = area
        self.group_column = group_column
        self.data_column = data_column
        self.data_filter = data_filter
        self.lag = lag
        self.days_to_predict = days_to_predict
        self.is_regional = is_regional
        self.pcm_data = pcm_data

    def get_region_pop(self, region, pop_df, prov_df):
        """
        Computes the total population for a region
        starting from the provinces' popuplation

        Parameters
        ----------
        region : str
            The region whose population we need

        pop_df : pandas DataFrame
            Data for provinces population

        prov_df : pandas DataFrame
            Data that associates each province with
            its region

        Returns
        -------
        N : int
            The population of the region
        """

        prov_list = prov_df[prov_df.Region == region]['Province'].values

        N = 0
        for prov in prov_list:
            N += pop_df.loc[
                (pop_df.Territorio == prov) &
                (pop_df.Eta == "Total")
                ]['Value'].values[0]

        return N

    def get_prov_pop(self):
        return self.pop_prov_df.loc[
            (self.pop_prov_df.Territorio == self.area) &
            (self.pop_prov_df.Eta == "Total")
            ]['Value'].values[0]

    def fix_arr(self, arr):
        arr[arr < 0] = 0
        arr[np.isinf(arr)] = 0
        return np.nan_to_num(arr)

    def lag_data(self, data, lag=7, return_all=False):
        if isinstance(data, np.ndarray):
            N = data.shape[0]
        else:
            N = len(data)

        X = np.empty(shape=(N-lag, lag+1))

        for i in range(lag, N):
            X[i-lag, ] = [data[i-j] for j in range(lag+1)]

        if not return_all:
            return X[-1, 1:]
        else:
            return X[:, 1:], X[:, 0]

    def _prepare_data_regional(self):
        data_df = self.data_df.loc[
            (self.data_df[self.group_column] == self.area),
            [
                'data',
                'totale_positivi',
                'dimessi_guariti',
                'deceduti', 'totale_casi',
                'nuovi_positivi'
            ]
        ]

        data_df = data_df.query(
            self.data_filter + ' > ' + self.data_column
        )

        data_df['suscettibili'] = self.pop - data_df['totale_casi']

        data_df = data_df.loc[:, [
            'data',
            'totale_positivi',
            'dimessi_guariti',
            'deceduti',
            'suscettibili',
            'nuovi_positivi'
        ]]

        return data_df

    def _prepare_data_provincial(self):
        regione = self.data_df[
            self.data_df[self.group_column] == self.area
        ]['Region'].values[0]

        pop = self.get_prov_pop()

        pcm_data = self.pcm_data.loc[
            (self.pcm_data['denominazione_regione'] == regione),
            [
                'data',
                'totale_positivi',
                'dimessi_guariti',
                'deceduti', 'totale_casi',
                'nuovi_positivi'
            ]].reset_index(drop=True)

        data_df = self.data_df.loc[
            (self.data_df[self.group_column] == self.area),
            [
                "Date",
                "New_cases",
                "Curr_pos_cases",
                "Tot_deaths"
            ]].reset_index(drop=True)

        recov_rate = (
            pcm_data['dimessi_guariti'] /
            pcm_data['totale_casi'])[:data_df.shape[0]]

        recov_rate = self.fix_arr(recov_rate)

        recov = recov_rate * data_df['Curr_pos_cases'].values
        data_df['dimessi_guariti'] = recov

        infected = data_df['Curr_pos_cases'].values - \
            data_df['Tot_deaths'].values - \
            data_df['dimessi_guariti']

        data_df['totale_positivi'] = infected

        data_df['suscettibili'] = pop - data_df['Curr_pos_cases']

        query = (
            pd.Timestamp(self.data_filter) +
            pd.DateOffset(self.days_to_predict)
            ).strftime('%Y%m%d') + ' > ' + self.data_column

        real_df = data_df.query(query)

        real_df.rename(
            columns={
                "New_cases": "nuovi_positivi",
                "Curr_pos_cases": "totale_casi",
                "Tot_deaths": "deceduti",
                "Date": "data"
            }, inplace=True)

        real_df = real_df.loc[:, [
            'data',
            'totale_positivi',
            'dimessi_guariti',
            'deceduti',
            'suscettibili',
            'nuovi_positivi'
        ]]

        self._realdf = real_df

        data_df = data_df.query(
            self.data_filter + ' > ' + self.data_column
        )

        data_df.rename(
            columns={
                "New_cases": "nuovi_positivi",
                "Curr_pos_cases": "totale_casi",
                "Tot_deaths": "deceduti",
                "Date": "data"
            }, inplace=True)

        data_df = data_df.astype({
            'totale_positivi': 'int32',
            'dimessi_guariti': 'int32',
            'deceduti': 'int32',
            'suscettibili': 'int32',
            'nuovi_positivi': 'int32'
        })

        data_df = data_df.loc[:, [
            'data',
            'totale_positivi',
            'dimessi_guariti',
            'deceduti',
            'suscettibili',
            'nuovi_positivi'
        ]]

        return data_df

    def fit(self):
        if self.is_regional:
            self.pop = self.get_region_pop(
                region=self.area,
                pop_df=self.pop_prov_df,
                prov_df=self.prov_list_df
            )

            data_df = self._prepare_data_regional()
        else:
            self.pop = self.get_prov_pop()
            data_df = self._prepare_data_provincial()

        n = data_df.shape[0]

        gamma = np.diff(data_df['dimessi_guariti'].values) / \
            data_df.iloc[:n-1]['totale_positivi'].values

        alpha = np.diff(data_df['deceduti'].values) / \
            data_df.iloc[:n-1]['totale_positivi'].values

        beta = (self.pop/data_df.iloc[:n-1]['suscettibili'].values) * \
            (np.diff(data_df['totale_positivi'].values) +
             np.diff(data_df['dimessi_guariti'].values) +
             np.diff(data_df['deceduti'].values)) / \
            data_df.iloc[:n-1]['totale_positivi'].values
        R0 = beta/(gamma+alpha)

        gamma = self.fix_arr(gamma)
        alpha = self.fix_arr(alpha)
        beta = self.fix_arr(beta)
        R0 = self.fix_arr(R0)

        reg_beta = LinearRegression().fit(
            *self.lag_data(beta, self.lag, True)
        )
        reg_gamma = LinearRegression().fit(
            *self.lag_data(gamma, self.lag, True)
        )
        reg_alpha = LinearRegression().fit(
            *self.lag_data(alpha, self.lag, True)
        )

        S = np.zeros(self.days_to_predict + 2)
        I = np.zeros(self.days_to_predict + 2)
        R = np.zeros(self.days_to_predict + 2)
        D = np.zeros(self.days_to_predict + 2)
        S[0] = data_df.iloc[-1]['suscettibili']
        I[0] = data_df.iloc[-1]['totale_positivi']
        R[0] = data_df.iloc[-1]['dimessi_guariti']
        D[0] = data_df.iloc[-1]['deceduti']

        for i in range(self.days_to_predict + 1):
            _beta = self.fix_arr(
                reg_beta.predict(
                    self.lag_data(beta, self.lag).reshape(1, -1)
                )
            )
            _gamma = self.fix_arr(
                reg_gamma.predict(
                    self.lag_data(gamma, self.lag).reshape(1, -1)
                )
            )
            _alpha = self.fix_arr(
                reg_alpha.predict(
                    self.lag_data(alpha, self.lag).reshape(1, -1)
                )
            )

            beta = np.append(beta, _beta, axis=0)
            gamma = np.append(gamma, _gamma, axis=0)
            alpha = np.append(alpha, _alpha, axis=0)

            dIdt = np.round((1 + _beta*(S[i]/self.pop) - _gamma - _alpha)*I[i])
            dRdt = np.round(R[i] + _gamma * I[i])
            dDdt = np.round(D[i] + _alpha * I[i])
            dSdt = self.pop-dIdt[0]-dRdt[0]-dDdt[0]

            S[i+1] = dSdt
            I[i+1] = dIdt
            R[i+1] = dRdt
            D[i+1] = dDdt

        S = S[1:]
        I = I[1:]
        R = R[1:]
        D = D[1:]

        dates = pd.date_range(
            start=(
                data_df.iloc[-1]['data'] + pd.DateOffset(1)
            ).strftime('%Y-%m-%d'),
            periods=self.days_to_predict + 1
        )

        tmp_df = pd.DataFrame(
            np.column_stack([
                np.zeros(self.days_to_predict + 1),
                I,
                R,
                D,
                S
            ]),
            columns=[
                'data',
                'totale_positivi',
                'dimessi_guariti',
                'deceduti',
                'suscettibili']
        )

        tmp_df['data'] = dates

        data_df = pd.concat([data_df, tmp_df], ignore_index=True)

        data_df['nuovi_positivi'] = [0] + list(
            np.diff(data_df['totale_positivi'].values) +
            np.diff(data_df['dimessi_guariti'].values) +
            np.diff(data_df['deceduti'].values))

        data_df['nuovi_positivi'] = \
            data_df['nuovi_positivi'].apply(lambda x: 0 if x < 0 else x)

        beta = np.append(beta, np.zeros((1,)), axis=0)
        gamma = np.append(gamma, np.zeros((1,)), axis=0)
        alpha = np.append(alpha, np.zeros((1,)), axis=0)

        data_df['beta'] = beta
        data_df['gamma'] = gamma
        data_df['alpha'] = alpha
        data_df['R0'] = self.fix_arr(beta/(gamma+alpha))
        data_df = data_df[:-1]

        data_df = data_df.astype({
            'totale_positivi': 'int32',
            'dimessi_guariti': 'int32',
            'deceduti': 'int32',
            'suscettibili': 'int32',
            'nuovi_positivi': 'int32'})

        self._fdf = data_df
        self._realdf = self._get_real_data()

        return data_df

    @property
    def fitted_df(self):
        return self._fdf

    @property
    def real_df(self):
        return self._realdf

    def _get_real_data(self):
        if not self.is_regional:
            return self._realdf
        else:
            real_df = self.data_df[
                self.data_df[self.group_column] == self.area
                ][[
                    'data',
                    'totale_positivi',
                    'dimessi_guariti',
                    'deceduti',
                    'totale_casi',
                    'nuovi_positivi'
                ]]

            query = (
                pd.Timestamp(self.data_filter) +
                pd.DateOffset(self.days_to_predict)
                ).strftime('%Y%m%d') + ' > data'

            real_df = real_df.query(query)
            real_df['suscettibili'] = self.pop - real_df['totale_casi']
            real_df = real_df[[
                'data',
                'totale_positivi',
                'dimessi_guariti',
                'deceduti',
                'suscettibili',
                'nuovi_positivi'
            ]]

            self._realdf = real_df

            return real_df

    def extract_ys(self, y_true, y_pred, compart):
        if y_true is None or y_pred is None:
            y_true = self.real_df[compart].values
            y_pred = self.fitted_df[compart].values

        return y_true, y_pred

    def mae(self, y_true=None, y_pred=None, compart=None):
        return mean_absolute_error(*self.extract_ys(y_true, y_pred, compart))

    def mse(self, y_true=None, y_pred=None, compart=None):
        return mean_squared_error(*self.extract_ys(y_true, y_pred, compart))

    def rmse(self, y_true=None, y_pred=None, compart=None):
        return mean_squared_error(*self.extract_ys(y_true, y_pred, compart),
                                  squared=False)
