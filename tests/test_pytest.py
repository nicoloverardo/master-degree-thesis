import pandas as pd
import numpy as np
import plotly.graph_objects
from ipywidgets import widgets
from src.utils import load_data, get_region_pop
from src.sird import beta, logistic_R0, sird, DeterministicSird
from src.fbp import ProphetModel
from src.plots import (
    general_plot,
    custom_plot,
    daily_main_indic_plot,
    plot_fbp_comp,
    inter_dropdown_plot,
)

import tensorflow as tf
from src.tfts import WindowGenerator


def test_data_loading():
    covidpro_df, dpc_regioni_df, dpc_province_df, pop_prov_df, prov_list_df = load_data(
        "data"
    )

    assert isinstance(covidpro_df, pd.DataFrame)
    assert isinstance(dpc_regioni_df, pd.DataFrame)
    assert isinstance(dpc_province_df, pd.DataFrame)
    assert isinstance(pop_prov_df, pd.DataFrame)
    assert isinstance(prov_list_df, pd.DataFrame)


def test_beta_R0():
    T = range(1, 10)
    betas = [beta(t=t, R_0_start=3, k=0.1, x0=20, R_0_end=1, gamma=1 / 7) for t in T]
    R0s = [logistic_R0(t=t, R_0_start=3, k=0.1, x0=20, R_0_end=1) for t in T]

    assert betas[0] >= betas[1]
    assert betas[-1] <= betas[-2]
    assert R0s[0] >= R0s[1]
    assert R0s[-1] <= R0s[-2]


def test_get_region_pop():
    _, _, _, pop_prov_df, prov_list_df = load_data("data")

    assert isinstance(int(get_region_pop("Piemonte", pop_prov_df, prov_list_df)), int)


def test_deterministic_sird():
    covidpro_df, dpc_regioni_df, _, pop_prov_df, prov_list_df = load_data("data")

    is_regional = False
    pcm_data = dpc_regioni_df
    group_column = "Province"
    data_df = covidpro_df
    data_column = "Date"
    lags = 7
    days_to_predict = 14
    data_filter = "20200630"
    area = "Firenze"

    model = DeterministicSird(
        data_df=data_df,
        pop_prov_df=pop_prov_df,
        prov_list_df=prov_list_df,
        area=area,
        group_column=group_column,
        data_column=data_column,
        data_filter=data_filter,
        lag=lags,
        days_to_predict=days_to_predict,
        is_regional=is_regional,
        pcm_data=pcm_data,
    )

    res = model.fit()

    assert isinstance(res, pd.DataFrame)
    assert isinstance(model.real_df, pd.DataFrame)


def test_general_plot():
    fig = general_plot(
        t=[1, 2],
        data=np.array([[1, 2], [3, 4]]),
        title="Test",
        output_image=False,
        template="plotly_white",
        output_figure=True,
    )

    assert isinstance(fig, plotly.graph_objects.Figure)


def test_custom_plot():
    _, dpc_regioni_df, _, _, _ = load_data("data")
    fig = custom_plot(
        df=dpc_regioni_df,
        ydata=["totale_casi"],
        title="Main indicators",
        xtitle="",
        ytitle="Individuals",
        group_column="denominazione_regione",
        area_name="Piemonte",
        template="plotly_white",
        show_title=False,
    )

    fig2 = inter_dropdown_plot(
        options=dpc_regioni_df.denominazione_regione.unique(),
        default_value="Lombardia",
        dropdown_label="Regione",
        y=["totale_casi"],
        legend_titles=["Total cases"],
        data=dpc_regioni_df,
        group_column="denominazione_regione",
        x="data",
        title="",
        xtitle="",
        ytitle="Individuals",
        output_image=False,
        blend_legend=False,
    )

    assert isinstance(fig, plotly.graph_objects.Figure)
    assert isinstance(fig2, widgets.VBox)


def test_daily_plot():
    _, dpc_regioni_df, _, _, _ = load_data("data")

    fig = daily_main_indic_plot(
        area="Piemonte",
        df=dpc_regioni_df,
        y_cols=["nuovi_positivi", "tamponi"],
        y_labels=["New positives", "tamponi"],
        output_figure=True,
        title="",
        template="plotly_white",
    )

    assert isinstance(fig, plotly.graph_objects.Figure)


def test_continuous_sird():
    _, _, _, pop_prov_df, _ = load_data("data")

    sirsol = sird(
        province="Firenze",
        pop_prov_df=pop_prov_df,
        gamma=1 / 7,
        alpha=0.1,
        R_0_start=3,
        k=0.1,
        x0=20,
        R_0_end=1,
    )

    assert isinstance(sirsol, np.ndarray)


def test_fbprophet():
    covidpro_df, _, _, _, _ = load_data("data")
    province = "Torino"
    compart = "New_cases"
    date = "Date"
    group_column = "Province"

    pm = ProphetModel(
        data=covidpro_df,
        area=province,
        compart=compart,
        group_column=group_column,
        date_column=date,
    )

    pm.fit()

    fig = plot_fbp_comp(df=pm.forecast, title="Forecast components", output_figure=True)

    assert len(pm.df) > 0
    assert len(pm.train) > 0
    assert len(pm.forecast) > 0
    assert pm.y_true.shape[0] == pm.y_true.shape[0]
    assert isinstance(fig, plotly.graph_objects.Figure)


def test_window_generator():
    df = pd.DataFrame(np.random.randint(0, 100, 100), columns=["Cases"])

    n = len(df)
    train_df = df[0 : int(n * 0.4)]
    val_df = df[int(n * 0.4) : int(n * 0.7)]
    test_df = df[int(n * 0.7) :]

    single_step_window = WindowGenerator(
        input_width=1,
        label_width=1,
        shift=1,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=["Cases"],
    )

    assert single_step_window.total_window_size == 2
    assert isinstance(single_step_window.train, tf.data.Dataset)
