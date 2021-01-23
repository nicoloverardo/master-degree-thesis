import pandas as pd
from src.utils import load_data


def test_data_loading():
    covidpro_df, dpc_regioni_df, dpc_province_df, pop_prov_df, prov_list_df = load_data(
        "data"
    )

    assert isinstance(covidpro_df, pd.DataFrame)
    assert isinstance(dpc_regioni_df, pd.DataFrame)
    assert isinstance(dpc_province_df, pd.DataFrame)
    assert isinstance(pop_prov_df, pd.DataFrame)
    assert isinstance(prov_list_df, pd.DataFrame)
