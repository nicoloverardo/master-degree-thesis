import numpy as np
import pandas as pd
import streamlit as st
import datetime

from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

from src.utils import load_data
from src.plots import (
    ac_plot,
    anomalies_plot,
    autocorr_indicators_plot,
    cross_corr_cases_plot,
    custom_plot,
    daily_main_indic_plot,
    data_for_plot,
    general_plot,
    plot_fbp_comp,
    plot_ts_decomp,
    plot_tstat_models,
    trend_corr_plot,
    discsid_param_plot,
)
from src.sird import DeterministicSird, sird
from src.ts import decompose_ts, adf_test_result, kpss_test_result
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

import tensorflow as tf
from src.tfts import (
    WindowGenerator,
    Baseline,
    compile_and_fit,
    plot_metrics,
    plot_comparison_results,
)


import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_PATH = "data"


@st.cache(show_spinner=False)
def load_df():
    return load_data(DATA_PATH)


@st.cache(show_spinner=False)
def load_icu_df():
    return pd.read_csv(Path(DATA_PATH, "icu.csv"))


@st.cache(show_spinner=False)
def compute_sird(
    prov,
    pop_prov_df,
    prov_list_df=None,
    r0_start=3.5,
    r0_end=0.9,
    k=0.9,
    x0=20,
    alpha=0.1,
    gamma=1 / 7,
):

    """Compute the continuous SIRD model version."""

    return sird(
        province=prov,
        pop_prov_df=pop_prov_df,
        prov_list_df=prov_list_df,
        gamma=gamma,
        alpha=alpha,
        R_0_start=r0_start,
        k=k,
        x0=x0,
        R_0_end=r0_end,
    )


@st.cache(show_spinner=False)
def data_sird_plot(df, column, comp_array, province_selectbox, is_regional):

    """Utility function that returns data useful for plots."""

    return data_for_plot(
        compart="Infected",
        df=df,
        column=column,
        comp_array=comp_array,
        province=province_selectbox,
        is_regional=is_regional,
    )


@st.cache(show_spinner=False)
def compute_daily_changes(df):
    """Gets daily variation of the regional indicators"""

    df["ricoverati_con_sintomi_giorno"] = df["ricoverati_con_sintomi"] - df[
        "ricoverati_con_sintomi"
    ].shift(1)

    df["terapia_intensiva_giorno"] = df["terapia_intensiva"] - df[
        "terapia_intensiva"
    ].shift(1)

    df["deceduti_giorno"] = df["deceduti"] - df["deceduti"].shift(1)

    df["tamponi_giorno"] = df["tamponi"] - df["tamponi"].shift(1)

    df["casi_testati_giorno"] = df["casi_testati"] - df["casi_testati"].shift(1)

    df["dimessi_guariti_giorno"] = df["dimessi_guariti"] - df["dimessi_guariti"].shift(
        1
    )

    df["isolamento_domiciliare_giorno"] = df["isolamento_domiciliare"] - df[
        "isolamento_domiciliare"
    ].shift(1)

    return df


@st.cache(show_spinner=False)
def compute_autocorr_df(df, days, is_regional=True):
    """
    Compute autocorrelations and cross-correlations
    of the main indicators
    """

    if not is_regional:
        pos_col = "New_cases"
        deaths_col = "Deaths"
    else:
        pos_col = "nuovi_positivi"
        deaths_col = "deceduti_giorno"

    data = pd.DataFrame(
        {
            "giorni": range(days),
            "autocor_nuovi_positivi": [
                df[pos_col].corr(df[pos_col].shift(i)) for i in range(days)
            ],
            "autocor_nuovi_decessi": [
                df[deaths_col].corr(df[deaths_col].shift(i)) for i in range(days)
            ],
            "crosscor_decessi_nuovi_positivi": [
                df[deaths_col]
                .rolling(7, center=True)
                .mean()
                .corr(df[pos_col].rolling(7, center=True).mean().shift(i))
                for i in range(days)
            ],
        }
    )

    if is_regional:
        data["autocor_tamponi_eseguiti"] = [
            df["tamponi_giorno"].corr(df["tamponi_giorno"].shift(i))
            for i in range(days)
        ]

        data["autocor_casi_testati"] = [
            df["casi_testati_giorno"].corr(df["casi_testati_giorno"].shift(i))
            for i in range(days)
        ]

        data["autocor_nuovi_ricoverati"] = [
            df["ricoverati_con_sintomi_giorno"].corr(
                df["ricoverati_con_sintomi_giorno"].shift(i)
            )
            for i in range(days)
        ]

        data["autocor_nuove_TI"] = [
            df["terapia_intensiva_giorno"].corr(df["terapia_intensiva_giorno"].shift(i))
            for i in range(days)
        ]

    return data


@st.cache(show_spinner=False)
def decompose_series(df, column):
    return decompose_ts(df, column)


@st.cache
def get_train_df(df, date, column):
    return df.query(date.strftime("%Y%m%d") + " >= " + column)


@st.cache
def get_test_df(df, date, column):
    return df.query(date.strftime("%Y%m%d") + " < " + column)


@st.cache(show_spinner=False)
def compute_acf(data, lags):
    return acf(data, nlags=lags, alpha=0.05)


@st.cache(show_spinner=False)
def compute_pacf(data, lags):
    return pacf(data, nlags=lags, alpha=0.05)


@st.cache(show_spinner=False)
def run_fbp(train, data_column, column, days_to_pred):
    train_df = train.reset_index()
    train_df = train_df.loc[:, [data_column, column]]
    train_df.columns = ["ds", "y"]

    m = Prophet(interval_width=0.95)
    m.add_country_holidays(country_name="IT")
    m.fit(train_df)

    future_df = m.make_future_dataframe(periods=days_to_pred)

    forecast = m.predict(future_df)

    return m, forecast


@st.cache(show_spinner=False)
def run_sarimax(train, test, order):
    model = SARIMAX(train, order=order)

    sarimax_res = model.fit()
    yhat = sarimax_res.get_forecast(steps=len(test))
    sarimax_sf = yhat.summary_frame()

    return sarimax_res.fittedvalues, sarimax_res.summary(), sarimax_sf


def load_homepage():
    """Homepage"""

    st.write(
        "Welcome to the interactive dashboard of my thesis "
        "for the MSc in Data Science and Economics at "
        "Università degli Studi di Milano."
    )
    st.header("The Application")
    st.write(
        "This application is a Streamlit dashboard that can be used "
        "to explore the work of my master degree thesis."
    )
    st.write(
        "There are currently four pages available in the application "
        "and they are described below. To navigate between pages, "
        "use the dropdown menu in the sidebar. To reveal the sidebar, "
        "click on the arrow ＞ at top-left corner."
    )
    st.subheader("🗺️ Data exploration")
    st.markdown("* This gives a general overview of the data with interactive plots.")
    st.subheader("📈 Time Series")
    st.markdown(
        "* This page allows you to see predictions made using time "
        "series models and the Prophet library."
    )
    st.subheader("👓 SIRD")
    st.markdown(
        "* This page allows you to see predictions made using "
        "stochastic and deterministic SIRD models with time-dependent "
        "parameters."
    )
    st.subheader("🗝️ TensorFlow")
    st.markdown(
        "* This page serves to show predictions made using "
        "neural networks (such as LSTM) implemented through "
        "TensorFlow."
    )

    st.write("")
    st.write("")

    st.write(
        "If you are on a wide screen, you can make the app fit the entire "
        "width of the page. In order to do this, click on the top-right "
        'hamburger menu icon ☰, then click on "Settings", '
        'check "Show app in wide mode" and finally click on "Save".'
    )


def load_ts_page(covidpro_df, dpc_regioni_df):
    """Time series analysis and forecast page"""

    # Sidebar setup
    st.sidebar.header("Options")
    area_radio = st.sidebar.radio(
        "Regional or provincial predictions:", ["Regional", "Provincial"], index=1
    )

    group_column = "denominazione_regione"
    data_df = dpc_regioni_df
    data_column = "data"
    column = "nuovi_positivi"

    if area_radio == "Regional":
        area_selectbox = st.sidebar.selectbox(
            "Region:",
            dpc_regioni_df.denominazione_regione.unique(),
            int((dpc_regioni_df.denominazione_regione == "Piemonte").argmax()),
            key="area_selectbox_reg",
        )
    else:
        area_selectbox = st.sidebar.selectbox(
            "Province:",
            covidpro_df.Province.unique(),
            int((covidpro_df.Province == "Firenze").argmax()),
            key="area_selectbox_prov",
        )

        group_column = "Province"
        data_df = covidpro_df
        data_column = "Date"
        column = "New_cases"

    last_avail_date = data_df.iloc[-1][data_column]

    # Date pickers
    start_date_ts = st.sidebar.date_input(
        "Start date",
        datetime.date(2020, 2, 24),
        datetime.date(2020, 2, 24),
        last_avail_date,
    )
    end_date_ts = st.sidebar.date_input(
        "End date",
        datetime.date(2020, 7, 1),
        datetime.date(2020, 2, 24),
        last_avail_date,
    )

    days_to_pred = st.sidebar.slider("Days to predict", 1, 30, 14)

    # Filter data
    df_filtered = data_df.query(
        end_date_ts.strftime("%Y%m%d")
        + " >= "
        + data_column
        + " >= "
        + start_date_ts.strftime("%Y%m%d")
    )
    df_final = df_filtered.loc[(df_filtered[group_column] == area_selectbox), :]

    df_date_idx = df_final.set_index(data_column)

    # Decomposition
    st.header("Series decomposition")

    with st.spinner("Decomposing series"):
        decomp_res = decompose_series(df_date_idx, column)

        st.plotly_chart(
            plot_ts_decomp(
                x_dates=df_date_idx.index,
                ts_true=df_date_idx[column],
                decomp_res=decomp_res,
                output_figure=True,
            ),
            use_container_width=True,
        )

    with st.beta_expander("Stationarity tests"):
        adf_res = adf_test_result(df_date_idx[column])
        kpss_res = kpss_test_result(df_date_idx[column])

        st.info(
            f"""
        ADF statistic: {adf_res[0]}

        p-value: {adf_res[1]}
        """
        )

        st.write("")

        st.info(
            f"""
        KPSS statistic: {kpss_res[0]}

        p-value: {kpss_res[1]}
        """
        )

    st.write("")
    st.write("")

    max_lags = int(df_date_idx.shape[0] / 2) - 1

    st.header("Auto-correlation")
    with st.spinner("Plotting ACF and PACF"):
        lags_start = max_lags if max_lags < 60 else 60

        acf_val, ci_acf = compute_acf(df_date_idx[column], lags_start)
        pacf_val, ci_pacf = compute_pacf(df_date_idx[column], lags_start)

        st.plotly_chart(
            ac_plot(acf_val, ci_acf, output_figure=True, title="Auto-correlation"),
            use_container_width=True,
        )

        st.plotly_chart(
            ac_plot(
                pacf_val, ci_pacf, output_figure=True, title="Partial auto-correlation"
            ),
            use_container_width=True,
        )

        p_value = adf_test_result(df_date_idx[column])[0]
        st.write("Dickey-Fuller: p={0:.5f}".format(p_value))

    st.write("")
    st.write("")

    st.header("Anomalies")
    st.plotly_chart(
        anomalies_plot(df_date_idx, column, 7, output_figure=True),
        use_container_width=True,
    )

    # Forecasting
    st.header("Forecasting")

    test_date = pd.to_datetime(end_date_ts) - pd.Timedelta(days=days_to_pred)

    train = get_train_df(df_date_idx, test_date, data_column)
    test = get_test_df(df_date_idx, test_date, data_column)

    # Exponential Smoothing
    st.subheader("Exponential Smoothing")

    with st.spinner("Training model"):
        es_model = ExponentialSmoothing(
            train[column].values, seasonal_periods=7, trend="add", seasonal="add"
        )

        es_res = es_model.fit()
        es_yhat = es_res.predict(start=0, end=len(test) - 1)

        st.plotly_chart(
            plot_tstat_models(
                df=df_date_idx,
                train=train,
                test=test,
                fitted_vals=es_res.fittedvalues,
                yhat=es_yhat,
                column=column,
                output_figure=True,
            ),
            use_container_width=True,
        )

        mae = mean_absolute_error(test[column], es_yhat)
        mse = mean_squared_error(test[column], es_yhat)
        rmse = mean_squared_error(test[column], es_yhat, squared=False)

        with st.beta_expander("Show training results"):
            st.text("AIC: " + str(es_res.aic))
            st.text("AICC: " + str(es_res.aicc))
            st.text("BIC: " + str(es_res.bic))
            st.text("k: " + str(es_res.k))
            st.text("")
            st.text("MAE: " + str(np.round(mae, 3)))
            st.text("MSE: " + str(np.round(mse, 3)))
            st.text("RMSE: " + str(np.round(rmse, 3)))
            st.text("SSE: " + str(es_res.sse))
            st.text("")
            st.text("")
            st.text(es_res.mle_retvals)
            st.text(es_res.summary())

    st.write("")
    st.write("")
    st.write("")

    # ARIMA
    st.subheader("ARIMA")

    col1_arima, col2_arima, col3_arima = st.beta_columns(3)
    lags_arima = col1_arima.slider("Lags (p)", 0, 15, 1)
    dod_arima = col2_arima.slider("Degree of differencing (d)", 0, 15, 1)
    window_arima = col3_arima.slider("Window (q)", 0, 30, 1)

    with st.spinner("Training model"):
        sarimax_fv, sarimax_summary, sarimax_sf = run_sarimax(
            train[column], test, (lags_arima, dod_arima, window_arima)
        )

        st.plotly_chart(
            plot_tstat_models(
                df=df_date_idx,
                train=train,
                test=test,
                fitted_vals=sarimax_fv,
                yhat=sarimax_sf,
                column=column,
                output_figure=True,
            ),
            use_container_width=True,
        )

        mae = mean_absolute_error(test[column], sarimax_sf["mean"])
        mse = mean_squared_error(test[column], sarimax_sf["mean"])
        rmse = mean_squared_error(test[column], sarimax_sf["mean"], squared=False)

        with st.beta_expander("Show training results"):
            st.text("MAE: " + str(np.round(mae, 3)))
            st.text("MSE: " + str(np.round(mse, 3)))
            st.text("RMSE: " + str(np.round(rmse, 3)))
            st.text("")
            st.text("")
            st.text(sarimax_summary)

    st.write("")
    st.write("")
    st.write("")

    # Prophet
    st.subheader("Facebook Prophet")

    with st.spinner("Training model"):
        m, forecast = run_fbp(train, data_column, column, days_to_pred)

        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(
            title="Forecast",
            yaxis_title="Number of individuals",
            template="plotly_white",
            title_x=0.5,
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.plotly_chart(
            plot_fbp_comp(df=forecast, title="Forecast components", output_figure=True),
            use_container_width=True,
        )

        test_df = test.reset_index()
        yhat_df = forecast.iloc[-days_to_pred:]
        mae = mean_absolute_error(test_df[column], yhat_df.yhat)
        mse = mean_squared_error(test_df[column], yhat_df.yhat)
        rmse = mean_squared_error(test_df[column], yhat_df.yhat, squared=False)

        with st.beta_expander("Show training results"):
            st.text("MAE: " + str(np.round(mae, 3)))
            st.text("MSE: " + str(np.round(mse, 3)))
            st.text("RMSE: " + str(np.round(rmse, 3)))

            st.dataframe(forecast)

    st.write("")
    st.write("")
    st.write("")
    st.subheader("🏗 Page under construction")


def load_tf_page(covidpro_df, dpc_regioni_df):
    """Neural Networks with TensorFlow page"""

    # Sidebar setup
    st.sidebar.header("Options")
    area_radio = st.sidebar.radio(
        "Regional or provincial predictions:", ["Regional", "Provincial"], index=1
    )

    group_column = "denominazione_regione"
    data_df = dpc_regioni_df
    data_column = "data"
    column = "nuovi_positivi"

    if area_radio == "Regional":
        area_selectbox = st.sidebar.selectbox(
            "Region:",
            dpc_regioni_df.denominazione_regione.unique(),
            int((dpc_regioni_df.denominazione_regione == "Piemonte").argmax()),
            key="area_selectbox_reg",
        )
    else:
        area_selectbox = st.sidebar.selectbox(
            "Province:",
            covidpro_df.Province.unique(),
            int((covidpro_df.Province == "Firenze").argmax()),
            key="area_selectbox_prov",
        )

        group_column = "Province"
        data_df = covidpro_df
        data_column = "Date"
        column = "New_cases"

    last_avail_date = data_df.iloc[-1][data_column]

    # Date pickers
    start_date_ts = st.sidebar.date_input(
        "Start date",
        datetime.date(2020, 2, 24),
        datetime.date(2020, 2, 24),
        last_avail_date,
    )
    end_date_ts = st.sidebar.date_input(
        "End date",
        datetime.date(2020, 7, 1),
        datetime.date(2020, 2, 24),
        last_avail_date,
    )

    # Filter data
    df_filtered = data_df.query(
        end_date_ts.strftime("%Y%m%d")
        + " >= "
        + data_column
        + " >= "
        + start_date_ts.strftime("%Y%m%d")
    )
    df_final = df_filtered.loc[(df_filtered[group_column] == area_selectbox), :]

    df_date_idx = df_final.set_index(data_column)
    df_date_idx = df_date_idx.loc[:, [column]]

    column_indices = {name: i for i, name in enumerate(df_date_idx.columns)}

    n = len(df_date_idx)
    train_df = df_date_idx[0 : int(n * 0.4)]
    val_df = df_date_idx[int(n * 0.4) : int(n * 0.7)]
    test_df = df_date_idx[int(n * 0.7) :]

    max_size = min(len(train_df), len(val_df), len(test_df))

    days_to_pred = st.sidebar.slider(
        "Input length/Days to predict", 1, max_size, max_size // 2
    )

    wide_window = WindowGenerator(
        input_width=days_to_pred,
        label_width=days_to_pred,
        shift=days_to_pred,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=[column],
    )

    val_performance = {}
    performance = {}

    st.header("Baseline")
    with st.spinner("Training model"):
        baseline = Baseline(label_index=column_indices[column])

        baseline.compile(
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError()],
        )

        st.pyplot(wide_window.plot(baseline), use_container_width=True)

        with st.beta_expander("Show training results"):
            val_performance["Baseline"] = baseline.evaluate(wide_window.val, verbose=0)
            performance["Baseline"] = baseline.evaluate(wide_window.test, verbose=0)

            st.text("Val. MAE: " + str(val_performance["Baseline"][1]))
            st.text("Test MAE: " + str(performance["Baseline"][1]))
            st.text("")
            st.text("Val. MSE: " + str(val_performance["Baseline"][2]))
            st.text("Test MSE: " + str(performance["Baseline"][2]))

    st.header("Dense")
    with st.spinner("Training model"):
        dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=64, activation="relu"),
                tf.keras.layers.Dense(units=64, activation="relu"),
                tf.keras.layers.Dense(units=1),
            ]
        )

        history = compile_and_fit(dense, wide_window)

        st.pyplot(wide_window.plot(dense), use_container_width=True)

        with st.beta_expander("Show training results"):
            val_performance["Dense"] = dense.evaluate(wide_window.val, verbose=0)
            performance["Dense"] = dense.evaluate(wide_window.test, verbose=0)

            st.text("Val. MAE: " + str(val_performance["Dense"][1]))
            st.text("Test MAE: " + str(performance["Dense"][1]))
            st.text("")
            st.text("Val. MSE: " + str(val_performance["Dense"][2]))
            st.text("Test MSE: " + str(performance["Dense"][2]))

            st.pyplot(plot_metrics(history))

    st.header("LSTM")
    with st.spinner("Training model"):
        lstm_model = tf.keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(32, return_sequences=True),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1),
            ]
        )

        history = compile_and_fit(lstm_model, wide_window)

        st.pyplot(wide_window.plot(lstm_model), use_container_width=True)

        with st.beta_expander("Show training results"):
            val_performance["LSTM"] = lstm_model.evaluate(wide_window.val, verbose=0)
            performance["LSTM"] = lstm_model.evaluate(wide_window.test, verbose=0)

            st.text("Val. MAE: " + str(val_performance["LSTM"][1]))
            st.text("Test MAE: " + str(performance["LSTM"][1]))
            st.text("")
            st.text("Val. MSE: " + str(val_performance["LSTM"][2]))
            st.text("Test MSE: " + str(performance["LSTM"][2]))

            st.pyplot(plot_metrics(history))

    st.header("Comparison")
    st.pyplot(
        plot_comparison_results(lstm_model.metrics_names, val_performance, performance)
    )

    st.subheader("🏗 Page under construction")


def load_sird_page(covidpro_df, dpc_regioni_df, pop_prov_df, prov_list_df):
    """Page of the SIRD model"""

    # Sidebar setup
    st.sidebar.header("Options")
    area_radio = st.sidebar.radio(
        "Regional or provincial predictions:", ["Regional", "Provincial"], index=1
    )

    is_regional = True
    pcm_data = None
    group_column = "denominazione_regione"
    data_df = dpc_regioni_df
    data_column = "data"
    prov_df = prov_list_df
    column = "totale_positivi"

    if area_radio == "Regional":
        area_selectbox = st.sidebar.selectbox(
            "Region:",
            dpc_regioni_df.denominazione_regione.unique(),
            int((dpc_regioni_df.denominazione_regione == "Piemonte").argmax()),
            key="area_selectbox_reg",
        )
    else:
        area_selectbox = st.sidebar.selectbox(
            "Province:",
            covidpro_df.Province.unique(),
            int((covidpro_df.Province == "Firenze").argmax()),
            key="area_selectbox_prov",
        )
        is_regional = False
        pcm_data = dpc_regioni_df
        group_column = "Province"
        data_df = covidpro_df
        data_column = "Date"
        prov_df = None
        column = "New_cases"

    # ---------------
    # Continuous SIRD
    # ---------------

    st.header("Continuous SIRD")

    # Sird parameters
    col1, col2, col3 = st.beta_columns(3)

    r0_start = col1.slider("R0 start", 1.0, 6.0, 2.0)
    r0_end = col1.slider("R0 end", 0.01, 3.5, 0.3)
    k_value = col2.slider("R0 decrease rate", 0.01, 1.0, 0.2)
    x0_value = col2.slider("Lockdown day", 0, 100, 40)
    alpha_value = col3.slider("Death rate", 0.001, 1.0, 0.01)
    gamma_value = col3.slider("Recovery rate", 0.001, 1.0, 1 / 7)

    # Compute SIRD
    with st.spinner("Training continuous SIRD"):
        sirsol = compute_sird(
            area_selectbox,
            pop_prov_df,
            prov_df,
            r0_start,
            r0_end,
            k_value,
            x0_value,
            alpha_value,
            gamma_value,
        )

        S, I, R, D = sirsol

        times = list(range(sirsol.shape[1]))

        # SIRD plot
        st.plotly_chart(
            general_plot(
                t=times,
                data=sirsol,
                title="SIRD model",
                traces_visibility=["legendonly"] + [True] * 3,
                output_image=False,
                template="plotly_white",
                output_figure=True,
                horiz_legend=True,
            ),
            use_container_width=True,
        )

        names, title, data, modes = data_sird_plot(
            data_df, column, I, area_selectbox, is_regional
        )

        # Comparison SIRD plot
        st.plotly_chart(
            general_plot(
                t=times,
                title="SIRD infected comparison",
                data=data,
                names=names,
                modes=modes,
                blend_legend=False,
                output_image=False,
                traces_visibility=["legendonly"] + [True] * 2,
                template="plotly_white",
                output_figure=True,
                horiz_legend=True,
            ),
            use_container_width=True,
        )

        # Show metrics
        mae = mean_absolute_error(data[1], data[2])
        mse = mean_squared_error(data[1], data[2])
        rmse = mean_squared_error(data[1], data[2], squared=False)

        with st.beta_expander("Show metrics"):
            st.info("MAE: " + str(np.round(mae, 3)))
            st.info("MSE: " + str(np.round(mse, 3)))
            st.info("RMSE: " + str(np.round(rmse, 3)))

    st.text("")
    st.text("")
    st.text("")

    # -------------
    # Discrete SIRD
    # -------------
    st.header("Discrete SIRD")

    (
        col1_reg,
        col2_reg,
    ) = st.beta_columns(2)

    lags = col1_reg.slider("Lags", 5, 15, 7)
    days_to_predict = col2_reg.slider("Days to predict", 5, 30, 14)
    data_filter = "20200630"

    # Define and fit SIRD
    with st.spinner("Training discrete SIRD"):
        model = DeterministicSird(
            data_df=data_df,
            pop_prov_df=pop_prov_df,
            prov_list_df=prov_list_df,
            area=area_selectbox,
            group_column=group_column,
            data_column=data_column,
            data_filter=data_filter,
            lag=lags,
            days_to_predict=days_to_predict,
            is_regional=is_regional,
            pcm_data=pcm_data,
        )

        res = model.fit()
        real_df = model.real_df

        # Infected
        st.plotly_chart(
            general_plot(
                t=real_df["data"],
                title="Daily infected of " + area_selectbox,
                data=[real_df["nuovi_positivi"].values, res["nuovi_positivi"].values],
                names=["Real", "Prediction"],
                modes=["markers", "lines"],
                blend_legend=False,
                output_image=False,
                output_figure=True,
                xtitle="",
                horiz_legend=True,
                template="plotly_white",
                prediction_size=days_to_predict,
            ),
            use_container_width=True,
        )

        # Deaths
        st.plotly_chart(
            general_plot(
                t=real_df["data"],
                title="Cumulative deaths of " + area_selectbox,
                data=[real_df["deceduti"].values, res["deceduti"].values],
                names=["Real", "Prediction"],
                modes=["markers", "lines"],
                blend_legend=False,
                output_image=False,
                output_figure=True,
                xtitle="",
                horiz_legend=True,
                template="plotly_white",
                prediction_size=days_to_predict,
                pred_label_top=False,
            ),
            use_container_width=True,
        )

        # Cumulative infected
        st.plotly_chart(
            general_plot(
                t=real_df["data"],
                title="Total positives of " + area_selectbox,
                data=[real_df["totale_positivi"].values, res["totale_positivi"].values],
                names=["Real", "Prediction"],
                modes=["markers", "lines"],
                blend_legend=False,
                output_image=False,
                output_figure=True,
                xtitle="",
                horiz_legend=True,
                template="plotly_white",
                prediction_size=days_to_predict,
            ),
            use_container_width=True,
        )

        # Params plot
        st.plotly_chart(
            discsid_param_plot(
                t=real_df["data"],
                title="Parameters",
                data=[
                    res["beta"].values,
                    res["gamma"].values,
                    res["alpha"].values,
                    res["R0"].values,
                ],
                names=["Beta", "Gamma", "Alpha", "R0"],
                modes=["lines"] * 4,
                blend_legend=False,
                output_image=False,
                output_figure=True,
                xtitle="",
                horiz_legend=True,
                template="plotly_white",
            ),
            use_container_width=True,
        )

        # Show metrics
        mae_tot_pos = model.mae(compart="totale_positivi")
        mse_tot_pos = model.mse(compart="totale_positivi")
        mae_deaths = model.mae(compart="deceduti")
        mse_deaths = model.mse(compart="deceduti")
        mae_rec = model.mae(compart="dimessi_guariti")
        mse_rec = model.mse(compart="dimessi_guariti")

        with st.beta_expander("Show metrics"):
            st.write(
                """
            The metrics MAE and MSE that you see below are 'averaged'
            because we first compute them for each of the three
            series in the plots above individually, and then we take
            the average of the three, since the model should be able to
            predict the number of positives and deaths at the same time.
            """
            )

            st.info(
                "Average MAE: "
                + str(np.round(np.mean([mae_tot_pos, mae_deaths, mae_rec]), 2))
            )
            st.info(
                "Average MSE: "
                + str(np.round(np.mean([mse_tot_pos, mse_deaths, mse_rec]), 2))
            )

    st.text("")
    st.text("")
    st.subheader("🏗 Page under construction")


def load_eda(covidpro_df, dpc_regioni_df, icu_df):
    """Explorative Data Analysis page"""

    st.sidebar.header("Options")

    # Date pickers
    start_date_eda = st.sidebar.date_input(
        "Start date",
        datetime.date(2020, 2, 24),
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]["data"],
    )
    end_date_eda = st.sidebar.date_input(
        "End date",
        datetime.date(2020, 7, 1),
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]["data"],
    )

    # Raw data checkbox
    show_raw_data = st.sidebar.checkbox("Show raw data")

    # Cite text
    st.sidebar.markdown(
        """<font size='1'><span style='color:grey'>*"Daily changes in the
        main indicators" and "Correlations and auto-correlations" plots are
        taken and adapted from Alberto Danese repo and can be found
        [here](https://github.com/albedan/covid-ts-ita).*</span></font>
        """,
        unsafe_allow_html=True,
    )

    # --------------
    # Regional plots
    # --------------

    st.header("Regional plots")

    # Combobox
    region_selectbox = st.selectbox(
        "Region:",
        dpc_regioni_df.denominazione_regione.unique(),
        int((dpc_regioni_df.denominazione_regione == "Lombardia").argmax()),
    )

    # Filter data
    dpc_reg_filtered = dpc_regioni_df.query(
        end_date_eda.strftime("%Y%m%d")
        + " >= data >= "
        + start_date_eda.strftime("%Y%m%d")
    )

    dpc_final = dpc_reg_filtered[
        dpc_reg_filtered.denominazione_regione == region_selectbox
    ]

    daily_df = compute_daily_changes(dpc_final)
    autocorr_df = compute_autocorr_df(daily_df, 30)

    icu = icu_df.loc[(icu_df.Regione == region_selectbox), "Posti"].values[0]

    if show_raw_data:
        st.subheader("Raw data")
        st.write("Regional data:")
        st.write(daily_df)

        st.write("Corr. and auto-corr. data:")
        st.write(autocorr_df)

    # Plots
    st.subheader("Main trendlines")

    # Absolute values
    st.plotly_chart(
        custom_plot(
            df=dpc_reg_filtered,
            ydata=[
                "totale_casi",
                "dimessi_guariti",
                "totale_positivi",
                "isolamento_domiciliare",
                "totale_ospedalizzati",
                "ricoverati_con_sintomi",
                "terapia_intensiva",
                "deceduti",
            ],
            title="Main indicators",
            xtitle="",
            ytitle="Individuals",
            group_column="denominazione_regione",
            area_name=region_selectbox,
            blend_legend=False,
            legend_titles=[
                "Total cases",
                "Total recovered",
                "Total positives",
                "Home quarantine",
                "Hospitalized",
                "Hospitalized with symptoms",
                "Intensive care",
                "Total deaths",
            ],
            template="plotly_white",
            show_title=False,
            horiz_legend=True,
        ),
        use_container_width=True,
    )

    # % values
    st.plotly_chart(
        custom_plot(
            df=dpc_reg_filtered,
            ydata=["IC_R", "Hosp_R", "NC_R", "NP_R"],
            title="",
            xtitle="",
            ytitle="Fraction",
            group_column="denominazione_regione",
            area_name=region_selectbox,
            blend_legend=True,
            legend_titles=[
                "IC over tot. cases",
                "Hospitalized over tot. cases",
                "Positives over tampons",
                "Positives over tot. positives",
            ],
            template="plotly_white",
            show_title=False,
            horiz_legend=True,
        ),
        use_container_width=True,
    )

    # icu
    st.plotly_chart(
        custom_plot(
            df=dpc_reg_filtered,
            ydata=["terapia_intensiva"],
            title="Intensive care occupancy vs capacity",
            xtitle="",
            ytitle="Individuals",
            group_column="denominazione_regione",
            area_name=region_selectbox,
            blend_legend=True,
            legend_titles=["Intesive care"],
            template="plotly_white",
            show_title=False,
            horiz_legend=True,
            icu=icu,
        ),
        use_container_width=True,
    )

    # Daily changes in the main indicators
    st.subheader("Daily changes in the main indicators")

    st.plotly_chart(
        daily_main_indic_plot(
            area=region_selectbox,
            df=daily_df,
            y_cols=[
                "ricoverati_con_sintomi_giorno",
                "terapia_intensiva_giorno",
                "nuovi_positivi",
                "tamponi_giorno",
                "casi_testati_giorno",
                "deceduti_giorno",
                "dimessi_guariti_giorno",
                "isolamento_domiciliare_giorno",
            ],
            y_labels=[
                "Hosp. with symptoms",
                "Intensive care",
                "New positives",
                "Tampons",
                "Tested cases",
                "Deaths",
                "Recovered",
                "Home quarantine",
            ],
            output_figure=True,
            title="",
            template="plotly_white",
        ),
        use_container_width=True,
    )

    st.subheader("Correlations and auto-correlations")

    # Auto-correlations
    st.plotly_chart(
        autocorr_indicators_plot(
            df=autocorr_df,
            x_col="giorni",
            y_cols=[
                "autocor_casi_testati",
                "autocor_tamponi_eseguiti",
                "autocor_nuovi_positivi",
                "autocor_nuovi_ricoverati",
                "autocor_nuove_TI",
                "autocor_nuovi_decessi",
            ],
            y_labels=[
                "Tested cases",
                "Tampons",
                "Positives",
                "Recovered",
                "Intensive care",
                "Deaths",
            ],
            output_figure=True,
            title="Auto-correlations",
            template="plotly_white",
        ),
        use_container_width=True,
    )

    # Cross-correlation
    st.plotly_chart(
        cross_corr_cases_plot(
            df=autocorr_df,
            template="plotly_white",
            title="Cross-correlation deaths -<br>new cases (rolling avg. 7d)",
            output_figure=True,
        ),
        use_container_width=True,
    )

    giorni_max_cor = autocorr_df["crosscor_decessi_nuovi_positivi"].idxmax()
    valore_max_cor = round(autocorr_df["crosscor_decessi_nuovi_positivi"].max(), 4)

    st.markdown(
        f"The highest correlation between deaths and "
        f"new positives is after *{giorni_max_cor} days* and "
        f"it is equal to *{valore_max_cor}* (where perfect correlation = 1)"
    )

    # Cross-correlation trend
    st.plotly_chart(
        trend_corr_plot(
            df=daily_df,
            days_max_corr=giorni_max_cor,
            y_cols=["nuovi_positivi", "deceduti_giorno"],
            y_labels=["Positives", "Deaths"],
            data_column="data",
            output_figure=True,
        ),
        use_container_width=True,
    )

    st.text("")
    st.text("")

    # ----------------
    # Provincial plots
    # ----------------
    st.header("Provincial plots")

    # Combobox
    province_selectbox = st.selectbox(
        "Province:",
        covidpro_df.Province.unique(),
        int((covidpro_df.Province == "Piacenza").argmax()),
    )

    # Filter data
    covidpro_filtered = covidpro_df.query(
        end_date_eda.strftime("%Y%m%d")
        + " >= Date >= "
        + start_date_eda.strftime("%Y%m%d")
    )

    covidpro_final = covidpro_filtered[covidpro_filtered.Province == province_selectbox]

    autocorr_df_prov = compute_autocorr_df(covidpro_final, 30, is_regional=False)

    if show_raw_data:
        st.subheader("Raw data")
        st.write("Provincial data:")
        st.write(covidpro_final)

    # Plots
    st.subheader("Main trendlines")

    # Absolute values
    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
            xdata="Date",
            ydata=["Deaths", "New_cases"],
            title="",
            xtitle="",
            ytitle="Individuals",
            group_column="Province",
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=["Deaths", "New cases"],
            template="plotly_white",
            show_title=False,
            horiz_legend=True,
        ),
        use_container_width=True,
    )

    # Cumulative values
    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
            xdata="Date",
            ydata=["Tot_deaths", "Curr_pos_cases"],
            title="",
            xtitle="",
            ytitle="Individuals",
            group_column="Province",
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=["Total deaths", "Total cases"],
            template="plotly_white",
            show_title=False,
            horiz_legend=True,
        ),
        use_container_width=True,
    )

    # % values
    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
            xdata="Date",
            ydata=["NP_R", "DR"],
            title="",
            xtitle="",
            ytitle="Fraction",
            group_column="Province",
            area_name=province_selectbox,
            blend_legend=True,
            legend_titles=["Positives over total cases", "Deaths over total cases"],
            template="plotly_white",
            show_title=False,
            horiz_legend=True,
        ),
        use_container_width=True,
    )

    st.subheader("Correlations and auto-correlations")

    # Auto-correlations
    st.plotly_chart(
        autocorr_indicators_plot(
            df=autocorr_df_prov,
            x_col="giorni",
            y_cols=["autocor_nuovi_positivi", "autocor_nuovi_decessi"],
            y_labels=["Positives", "Deaths"],
            output_figure=True,
            title="Auto-correlations",
            template="plotly_white",
        ),
        use_container_width=True,
    )

    # Cross-correlations
    st.plotly_chart(
        cross_corr_cases_plot(
            df=autocorr_df_prov,
            template="plotly_white",
            title="Cross-correlation deaths -<br>new cases (rolling avg. 7d)",
            output_figure=True,
        ),
        use_container_width=True,
    )

    giorni_max_cor_prov = autocorr_df_prov["crosscor_decessi_nuovi_positivi"].idxmax()
    val_max_cor_prov = round(
        autocorr_df_prov["crosscor_decessi_nuovi_positivi"].max(), 4
    )

    st.markdown(
        f"The highest correlation between deaths and "
        f"new positives is after *{giorni_max_cor_prov} days* and "
        f"it is equal to *{val_max_cor_prov}* (where perfect correlation = 1)"
    )

    # Cross-correlation trend
    st.plotly_chart(
        trend_corr_plot(
            df=covidpro_final,
            days_max_corr=giorni_max_cor_prov,
            y_cols=["New_cases", "Deaths"],
            y_labels=["Positives", "Deaths"],
            data_column="Date",
            output_figure=True,
        ),
        use_container_width=True,
    )


def main():
    """Main routine of the app"""

    st.set_page_config(
        page_title="Master Degree Thesis - Verardo", page_icon="🎓", layout="centered"
    )

    st.title(
        "A dashboard for COVID-19 prediction in Italy "
        "at regional and provincial level"
    )

    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox(
        "Please select a page",
        ["Homepage", "Data Exploration", "Time Series", "SIRD", "TensorFlow"],
    )

    with st.spinner("Loading data"):
        covidpro_df, dpc_regioni_df, _, pop_prov_df, prov_list_df = load_df()
        icu_df = load_icu_df()

    if app_mode == "Homepage":
        load_homepage()
    elif app_mode == "Data Exploration":
        load_eda(covidpro_df, dpc_regioni_df, icu_df)
    elif app_mode == "Time Series":
        load_ts_page(covidpro_df, dpc_regioni_df)
    elif app_mode == "SIRD":
        load_sird_page(covidpro_df, dpc_regioni_df, pop_prov_df, prov_list_df)
    elif app_mode == "TensorFlow":
        load_tf_page(covidpro_df, dpc_regioni_df)


if __name__ == "__main__":
    main()
