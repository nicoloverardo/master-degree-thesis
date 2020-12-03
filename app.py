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
def load_data():
    covidpro_df = pd.read_csv(Path(DATA_PATH, CSVName.COVIDPRO_CSV))
    dpc_regioni_df = pd.read_csv(Path(DATA_PATH, CSVName.DPC_REGIONI))
    dpc_province_df = pd.read_csv(Path(DATA_PATH, CSVName.DPC_PROVINCE))
    pop_prov_df = pd.read_csv(Path(DATA_PATH, CSVName.POP_PROV_CSV))

    dpc_province_df['data'] = pd.to_datetime(dpc_province_df['data'])
    dpc_regioni_df['data'] = pd.to_datetime(dpc_regioni_df['data'])
    covidpro_df['Date'] = pd.to_datetime(covidpro_df['Date'])

    dpc_province_df.denominazione_provincia = \
    dpc_province_df.denominazione_provincia.str \
        .replace("Forlì-Cesena", "Forli-Cesena")

    dpc_regioni_df.denominazione_regione = \
        dpc_regioni_df.denominazione_regione.str \
            .replace("P.A. Trento", "Trentino Alto Adige") \
            .replace("P.A. Bolzano", "Trentino Alto Adige")
    
    covidpro_df.Region = covidpro_df.Region.str \
        .replace("P.A. Trento", "Trentino Alto Adige") \
        .replace("P.A. Bolzano", "Trentino Alto Adige")
    
    pop_prov_df.Territorio = pop_prov_df.Territorio.str \
        .replace("Valle d'Aosta", "Aosta") \
        .replace("Forlì-Cesena", "Forli-Cesena") \
        .replace("Massa-Carrara", "Massa Carrara") \
        .replace("L'Aquila", "L Aquila") \
        .replace("Reggio nell'Emilia", 'Reggio nell Emilia')
    
    dpc_regioni_df['NC_R'] = dpc_regioni_df['nuovi_positivi']/dpc_regioni_df['tamponi']
    dpc_regioni_df['NP_R'] = dpc_regioni_df['nuovi_positivi']/dpc_regioni_df['totale_positivi']
    dpc_regioni_df['IC_R'] = dpc_regioni_df['terapia_intensiva']/dpc_regioni_df['totale_positivi']
    dpc_regioni_df['Hosp_R'] = dpc_regioni_df['totale_ospedalizzati']/dpc_regioni_df['totale_positivi']
    dpc_regioni_df['DR'] = dpc_regioni_df['deceduti']/dpc_regioni_df['totale_positivi']

    covidpro_df['New_cases'] = covidpro_df['New_cases'].apply(lambda x: 0 if x is np.NaN or x < 0 else x)
    covidpro_df['Deaths'] = covidpro_df['Deaths'].apply(lambda x: 0 if x is np.NaN or x < 0 else x)

    covidpro_df['NP_R'] = covidpro_df.apply(compute_ratio_NP, axis=1)
    covidpro_df['DR'] = covidpro_df.apply(compute_ratio_DR, axis=1)

    dpc_regioni_df['NC_R_Rolling'] = dpc_regioni_df['NC_R'].rolling(window=7).mean()
    dpc_regioni_df['IC_R_Rolling'] = dpc_regioni_df['IC_R'].rolling(window=7).mean()
    dpc_regioni_df['totale_positivi_Rolling'] = dpc_regioni_df['totale_positivi'].rolling(window=7).mean()

    covidpro_df['NP_R_Rolling'] = covidpro_df['NP_R'].rolling(window=7).mean()
    covidpro_df['DR_Rolling'] = covidpro_df['DR'].rolling(window=7).mean()

    covidpro_df.fillna(0, inplace=True)
    dpc_regioni_df.fillna(0, inplace=True)
    dpc_province_df.fillna(0, inplace=True)

    return covidpro_df, dpc_regioni_df, dpc_province_df, pop_prov_df

def compute_ratio_NP(x):
    if x['Curr_pos_cases'] == 0:
        return 0
    else:
        return x['New_cases']/x['Curr_pos_cases']

def compute_ratio_DR(x):
    if x['Curr_pos_cases'] == 0:
        return 0
    else:
        return x['Deaths']/x['Curr_pos_cases']


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

    covidpro_df, dpc_regioni_df, dpc_province_df, pop_prov_df = load_data()

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

    data_load_state = st.text('Loading data...')

    st.plotly_chart(
        custom_plot(
            df=dpc_regioni_df,
            ydata=['totale_casi', 'totale_positivi', 'deceduti', 'totale_ospedalizzati', 'terapia_intensiva'],
            title='COVID-19 trendlines of ',
            xtitle='Data',
            ytitle='Unità',
            group_column='denominazione_regione',
            area_name='Lombardia',
            legend_titles=['Totale Casi', 'Totale Positivi', 'Deceduti', 'Totale Ospedalizzati', 'Terapia Intensiva']
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
            legend_titles=['Intensive care over total cases', 'Hospitalized over total cases', 'Positives over tampons rolling average', 'Intensive care over total cases rolling average', 'Positives over tampons', 'Positives over total positives'],
            blend_legend=True
        ),
    use_container_width=True)

    data_load_state.empty()

if __name__ == "__main__":
    main()