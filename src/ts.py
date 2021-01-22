import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from sklearn.metrics import mean_absolute_error

import plotly.graph_objects as go
from IPython.display import Image


def visualize_ts(df, date, column, province):
    plt.figure(figsize=(12, 5))
    plt.plot(df[date], df[column])
    plt.gca().set(title=column + " " + province, xlabel="Date", ylabel="Individuals")
    plt.show()


def show_boxplot(df, column):
    plt.figure(figsize=(8, 5))
    df.boxplot(column=[column], grid=False)
    plt.show()


def decompose_ts(df, column, model="additive"):
    return seasonal_decompose(df[column], model=model, extrapolate_trend="freq")


def _plot_decomp(result):
    result.plot()
    plt.gcf().set_size_inches(12, 10)
    plt.show()


def plot_decomposition(df=None, column=None, result=None):
    if result is None:
        if df is not None and column is not None:
            result = decompose_ts(df, column)
        else:
            raise ValueError("Provide result or df and column name")

    _plot_decomp(result)


def adf_test(data):
    """
    ADF test: null hypothesis is the time series
    possesses a unit root and is non-stationary
    """

    result = adfuller(data, autolag="AIC")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[4].items():
        print("Critical Values:")
        print(f"   {key}, {value}")


def adf_test_result(data):
    return adfuller(data, autolag="AIC")


def kpss_test_result(data):
    return kpss(data, regression="c")


def kpss_test(data):
    """
    KPSS test: opposite of ADF
    """

    result = kpss(data, regression="c")
    print("\nKPSS Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    for key, value in result[3].items():
        print("Critical Values:")
        print(f"   {key}, {value}")


def remove_trend(df, column):
    return df[column].values - decompose_ts(df, column).trend


def deseason_trend(df, column):
    return df[column].values / decompose_ts(df, column).seasonal


def plot_detrended_deseason(data, province, column, title):
    plt.figure(figsize=(12, 5))
    plt.plot(data)
    plt.title(title + " " + column + " of " + province, fontsize=16)
    plt.show()


def plot_detrended_deseason_plotly(
    data,
    title=None,
    output_image=True,
    output_figure=False,
    width=800,
    height=400,
    scale=2,
    ytitle="Number of individuals",
    template="plotly_white",
):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values))

    fig.update_layout(
        title=title,
        yaxis_title=ytitle,
        template=template,
        title_x=0.5,
    )

    if output_image:
        return Image(
            fig.to_image(format="png", width=width, height=height, scale=scale)
        )

    if output_figure:
        return fig

    return fig.show()


def plot_deseason_ma(df, column, province, windows=[7, 30]):
    plt.figure(figsize=(12, 5))

    plt.plot(df.index, df[column].values, label="Real")
    for w in windows:
        plt.plot(
            df.index,
            df[column].rolling(window=w).mean(),
            label="Moving avg. " + str(w) + " days",
        )

    plt.title("Deseasonalized new cases of " + province, fontsize=16)
    plt.legend()
    plt.show()


def plot_autocorr(df, column):
    autocorrelation_plot(df[column].tolist())
    plt.gcf().set_size_inches(9, 5)
    plt.show()


def plot_acf_pacf(df, column, lags=50):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
    plot_acf(df[column].tolist(), lags=50, ax=axes[0])
    plot_pacf(df[column].tolist(), lags=50, ax=axes[1])
    plt.show()


def plot_lag_plots(df, column):
    """
    Lag plots:

    If points get wide and scattered with increasing lag,
    this means lesser correlation
    """

    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:4]):
        lag_plot(df[column], lag=i + 1, ax=ax)
        ax.set_title("Lag " + str(i + 1))
    plt.show()


def comp_loess_df(df, column, percentage):
    return pd.DataFrame(
        lowess(df[column], np.arange(len(df[column])), frac=percentage)[:, 1],
        index=df.index,
        columns=[column],
    )


def plot_smoothing(df, column):
    # 1. Moving Average
    df_ma1 = df[column].rolling(7, center=True, closed="both").mean()
    df_ma2 = df[column].rolling(30, center=True, closed="both").mean()

    # 2. Loess Smoothing (5% and 15%)
    df_loess_5 = comp_loess_df(df, column, 0.05)
    df_loess_15 = comp_loess_df(df, column, 0.15)

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True, dpi=120)
    df[column].plot(ax=axes[0], color="k", title="Original Series")
    df_loess_5[column].plot(ax=axes[1], title="Loess Smoothed 5%")
    df_loess_15[column].plot(ax=axes[2], title="Loess Smoothed 15%")
    df_ma1.plot(ax=axes[3], title="Moving average (7)")
    df_ma2.plot(ax=axes[4], title="Moving average (30)")
    fig.suptitle("Smoothing", y=0.95, fontsize=14)
    plt.show()


def tsplot(y, lags=50, figsize=(12, 7)):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    p_value = adfuller(y)[1]
    ts_ax.set_title(
        "Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}".format(p_value)
    )
    plot_acf(y, lags=lags, ax=acf_ax)
    plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()
    plt.show()


def ApEn(U, m, r):
    """Compute Aproximate entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m + 1) - _phi(m))


def SampEn(U, m, r):
    """Compute Sample entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r])
            for i in range(len(x))
        ]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))


def anom_plot(
    df,
    compart,
    window,
    plot_intervals=False,
    scale=1.96,
    plot_anomalies=False,
    show_anomalies_label=False,
    legend_position="upper left",
    output_figure=False,
):

    rolling_mean = df[[compart]].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.plot(rolling_mean, "g", label="Rolling mean trend")

    if plot_intervals:
        mae = mean_absolute_error(df[[compart]][window:], rolling_mean[window:])

        deviation = np.std(df[[compart]][window:] - rolling_mean[window:])

        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)

        ax.plot(upper_bound, "r--", label="Upper Bound / Lower Bound")
        ax.plot(lower_bound, "r--")

        if plot_anomalies:
            anomalies = pd.DataFrame(
                index=df[[compart]].index, columns=df[[compart]].columns
            )

            anomalies[df[[compart]] < lower_bound] = df[[compart]][
                df[[compart]] < lower_bound
            ]

            anomalies[df[[compart]] > upper_bound] = df[[compart]][
                df[[compart]] > upper_bound
            ]

            ax.plot(anomalies, "ro", markersize=10)

            xmin, xmax = plt.xlim()
            ax.hlines(
                y=0, xmin=xmin, xmax=xmax, linestyles="dashed", colors="grey", alpha=0.3
            )

            if show_anomalies_label:
                ymin, ymax = plt.ylim()

                ax.vlines(
                    anomalies.dropna().index,
                    ymin=ymin,
                    ymax=ymax,
                    linestyles="dashed",
                    colors="grey",
                )

                for x in anomalies.dropna().index:
                    ax.text(
                        x,
                        ymin + 20,
                        x.strftime("%m-%d"),
                        rotation=90,
                        verticalalignment="center",
                    )

    ax.plot(df[[compart]][window:], label="Actual values")
    ax.legend(loc=legend_position)

    if output_figure:
        return fig

    plt.show()
