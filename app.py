import streamlit as st
import pandas as pd
import numpy as np

from src.utils import *
from src.plots import *
from src.sird import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_PATH = "data"

@st.cache
def load_df():
    return load_data(DATA_PATH)

def main():
    st.set_page_config(
        page_title='Master Degree Thesis - Verardo',
        page_icon='🎓',
        layout='centered')

    st.title('Master Degree Thesis')

    """
    Welcome to the interactive dashboard of my thesis
    for the MSc in Data Science and Economics at 
    Università degli Studi di Milano.
    """

    st.sidebar.title('Welcome')

    data_load_state = st.text('Loading data...')

    covidpro_df, dpc_regioni_df, dpc_province_df, pop_prov_df, prov_list_df = load_df()

    data_load_state.empty()

    if st.sidebar.checkbox('Show raw data'):
        st.header("Raw data")

        st.subheader("COVIDPro")
        covidpro_df[:100]

        st.subheader("PCM regions")
        dpc_regioni_df[:100]

        st.subheader("PCM provinces")
        dpc_province_df[:100]

    st.header("Regional plots")

    data_load_state = st.text('Plotting data...')

    st.plotly_chart(
        custom_plot(
            df=dpc_regioni_df,
            ydata=[
                'totale_casi',
                'totale_positivi',
                'deceduti',
                'totale_ospedalizzati',
                'terapia_intensiva'],
            title='COVID-19 trendlines of ',
            xtitle='Data',
            ytitle='Unità',
            group_column='denominazione_regione',
            area_name='Lombardia',
            legend_titles=[
                'Totale Casi',
                'Totale Positivi',
                'Deceduti',
                'Totale Ospedalizzati',
                'Terapia Intensiva']
        ),
    use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=dpc_regioni_df,
            ydata=['IC_R', 'Hosp_R', 'NC_R_Rolling', 'IC_R_Rolling', 'NC_R', 'NP_R'],
            title='COVID-19 trendlines of ',
            xtitle='Data',
            ytitle='Unità',
            group_column='denominazione_regione',
            area_name='Lombardia',
            blend_legend=True,
            legend_titles=[
                'Intensive care over total cases',
                'Hospitalized over total cases',
                'Positives over tampons rolling average',
                'Intensive care over total cases rolling average',
                'Positives over tampons',
                'Positives over total positives']    
        ),
    use_container_width=True)

    data_load_state.empty()

if __name__ == "__main__":
    main()