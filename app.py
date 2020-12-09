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

    covidpro_df, dpc_regioni_df, _, _, _ = load_df()

    data_load_state.empty()

    show_raw_data = st.sidebar.checkbox('Show raw data')

    st.header("Regional plots")

    region_selectbox = st.selectbox(
        "Region:",
        dpc_regioni_df.denominazione_regione.unique(),
        int((dpc_regioni_df.denominazione_regione == 'Lombardia').argmax())
    )

    if show_raw_data:
        dpc_regioni_df[dpc_regioni_df.denominazione_regione == region_selectbox]  # nopep8

    st.plotly_chart(
        custom_plot(
            df=dpc_regioni_df,
            ydata=[
                'totale_casi',
                'totale_positivi',
                'dimessi_guariti',
                'isolamento_domiciliare',
                'totale_ospedalizzati',
                'ricoverati_con_sintomi',
                'terapia_intensiva',
                'deceduti'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Individuals',
            group_column='denominazione_regione',
            area_name=region_selectbox,
            blend_legend=False,
            legend_titles=[
                'Totale Casi',
                'Totale Positivi',
                'Totale Guariti',
                'Isolamento Domiciliare',
                'Totale Ospedalizzati',
                'Ricoverati con sintomi',
                'Terapia Intensiva',
                'Deceduti'],
            template='simple_white'
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=dpc_regioni_df,
            ydata=[
                'IC_R', 'Hosp_R',
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
                'Positives over tampons',
                'Positives over tot. positives'],
            template='simple_white'
        ), use_container_width=True)

    st.header("Provincial plots")

    province_selectbox = st.selectbox(
        "Region:",
        covidpro_df.Province.unique(),
        int((covidpro_df.Province == 'Piacenza').argmax())
    )

    if show_raw_data:
        covidpro_df[covidpro_df.Province == province_selectbox]

    st.plotly_chart(
        custom_plot(
            df=covidpro_df,
            xdata='Date',
            ydata=[
                'Deaths', 'New_cases'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Individuals',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=[
                'Deceduti', 'Nuovi casi'],
            template='simple_white'
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=covidpro_df,
            xdata='Date',
            ydata=[
                'Tot_deaths', 'Curr_pos_cases'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Individuals',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=[
                'Totale deceduti', 'Totale positivi'],
            template='simple_white'
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=covidpro_df,
            xdata='Date',
            ydata=['NP_R', 'DR'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Fraction',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=True,
            legend_titles=[
                'Positives over total cases',
                'Deaths over total cases'
                ],
            template='simple_white'
        ), use_container_width=True)


if __name__ == "__main__":
    main()
