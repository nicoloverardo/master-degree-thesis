import pandas as pd
from src.utils import load_data
from src.sird import beta, logistic_R0
from src.fbp import ProphetModel


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
    betas = [beta(t=t, R_0_start=3, k=0.1, x0=20, R_0_end=1, gamma=1/7) for t in T]
    R0s = [logistic_R0(t=t, R_0_start=3, k=0.1, x0=20, R_0_end=1) for t in T]

    assert betas[0] >= betas[1]
    assert betas[-1] <= betas[-2]
    assert R0s[0] >= R0s[1]
    assert R0s[-1] <= R0s[-2]


def test_fbprophet():
    covidpro_df, _, _, _, _ = load_data("data")
    province = "Torino"
    compart = "New_cases"
    date = 'Date'
    group_column = 'Province'

    pm = ProphetModel(
        data=covidpro_df,
        area=province,
        compart=compart,
        group_column=group_column,
        date_column=date)

    pm.fit()

    assert len(pm.df) > 0
    assert len(pm.train) > 0
    assert len(pm.forecast) > 0
    assert pm.y_true.shape[0] == pm.y_true.shape[0]
