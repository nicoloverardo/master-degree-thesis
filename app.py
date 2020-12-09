import streamlit as st

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
    """Main routine of the app"""
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

    covidpro_df, dpc_regioni_df, dpc_province_df, \
        pop_prov_df, prov_list_df = load_df()

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

    region_selectbox = st.selectbox(
        "Region:",
        dpc_regioni_df.denominazione_regione.unique()
    )

    st.plotly_chart(
        custom_plot(
            df=dpc_regioni_df,
            ydata=[
                'totale_casi',
                'totale_positivi',
                'deceduti',
                'totale_ospedalizzati',
                'terapia_intensiva'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Individuals',
            group_column='denominazione_regione',
            area_name=region_selectbox,
            blend_legend=False,
            legend_titles=[
                'Totale Casi',
                'Totale Positivi',
                'Deceduti',
                'Totale Ospedalizzati',
                'Terapia Intensiva'],
            template='simple_white'
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=dpc_regioni_df,
            ydata=[
                'IC_R', 'Hosp_R',
                'NC_R_Rolling', 'IC_R_Rolling',
                'NC_R', 'NP_R'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Fraction',
            group_column='denominazione_regione',
            area_name=region_selectbox,
            blend_legend=True,
            legend_titles=[
                'IC over tot. cases',
                'Hospitalized over tot. cases',
                'Positives over tampons (rolling avg.)',
                'IC over tot. cases (rolling avg.)',
                'Positives over tampons',
                'Positives over tot. positives'],
            template='simple_white'
        ), use_container_width=True)

    st.header("Provincial plots")

    province_selectbox = st.selectbox(
        "Region:",
        covidpro_df.Province.unique()
    )

    st.plotly_chart(
        custom_plot(
            df=covidpro_df,
            xdata='Date',
            ydata=[
                'Deaths', 'New_cases',
                'Tot_deaths', 'Curr_pos_cases'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Individuals',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=[
                'Deceduti', 'Nuovi casi',
                'Tot. Morti', 'Totale positivi'],
            template='simple_white'
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=covidpro_df,
            xdata='Date',
            ydata=[
                'NP_R', 'NP_R_Rolling',
                'DR', 'DR_Rolling'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Fraction',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=True,
            legend_titles=[
                'Positives over total cases',
                'Positives over total cases rolling',
                'Deaths over total cases',
                'Deaths over total cases rolling'
                ],
            template='simple_white'
        ), use_container_width=True)


if __name__ == "__main__":
    main()
