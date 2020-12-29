import streamlit as st
import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error

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

    # Title
    st.title('A comparison of predictive models for COVID-19 in Italy')

    # Sidebar title
    st.sidebar.title('Menu')
    app_mode = st.sidebar.selectbox(
        "Please select a page", [
            "Homepage",
            "Data Exploration",
            "Time series",
            "SIRD model",
            "TensorFlow model"]
    )

    # Loading state label
    data_load_state = st.text('Loading data...')

    # Load data
    covidpro_df, dpc_regioni_df, _, pop_prov_df, prov_list_df = load_df()

    data_load_state.empty()

    if app_mode == 'Homepage':
        load_homepage()
    elif app_mode == 'Data Exploration':
        load_eda(covidpro_df, dpc_regioni_df)
    elif app_mode == 'Time series':
        load_ts_page(covidpro_df, dpc_regioni_df)
    elif app_mode == 'SIRD model':
        load_sird_page(covidpro_df, dpc_regioni_df, pop_prov_df, prov_list_df)
    elif app_mode == 'TensorFlow model':
        load_tf_page(covidpro_df, dpc_regioni_df)


def load_homepage():
    st.write(
        "Welcome to the interactive dashboard of my thesis "
        "for the MSc in Data Science and Economics at "
        "Università degli Studi di Milano."
    )
    st.header("The Application")
    st.write("This application is a Streamlit dashboard that can be used "
             "to explore the work of my master degree thesis.")
    st.write("There are currently four pages available in the application:")
    st.subheader("🧭 Data exploration")
    st.markdown("* This gives a general overview of the data with interactive "
                "plots.")
    st.subheader("📈 Time series")
    st.markdown("* This page allows you to see predictions made using time "
                "series models and the Prophet library.")
    st.subheader("👥 SIRD model")
    st.markdown("* This page allows you to see predictions made using "
                "stochastic and deterministic sird with time-dependent "
                "parameters.")
    st.subheader("🪄 TensorFlow model")
    st.markdown("* This page serves to show predictions made using "
                "neural networks (such as LSTM) implemented using "
                "TensorFlow.")


def load_ts_page(covidpro_df, dpc_regioni_df):
    st.subheader("🚧 Page under construction")


def load_sird_page(covidpro_df, dpc_regioni_df, pop_prov_df, prov_list_df):
    st.header("Continuous SIRD")
    province_selectbox = st.selectbox(
        "Province:",
        covidpro_df.Province.unique(),
        int((covidpro_df.Province == 'Firenze').argmax())
    )

    sirsol = sird(province_selectbox, pop_prov_df)
    S, I, R, D = sirsol

    times = list(range(sirsol.shape[1]))

    st.plotly_chart(
        general_plot(
            t=times,
            data=sirsol,
            title='SIRD ' + province_selectbox,
            traces_visibility=['legendonly'] + [True]*3,
            output_image=False,
            template='simple_white',
            output_figure=True
        ), use_container_width=True
    )

    names, title, data, modes = data_for_plot(
        'Infected',
        covidpro_df,
        'New_cases',
        I,
        province_selectbox
    )

    st.plotly_chart(
        general_plot(
            t=times,
            title=title,
            data=data,
            names=names,
            modes=modes,
            blend_legend=False,
            output_image=False,
            traces_visibility=['legendonly'] + [True]*2,
            template='simple_white',
            output_figure=True
        ), use_container_width=True
    )

    st.write("MAE: " + str(np.round(mean_absolute_error(data[1], data[2]), 3)))
    st.write("MSE: " + str(np.round(mean_squared_error(data[1], data[2]), 3)))
    st.write(
        "RMSE: " +
        str(np.round(mean_squared_error(data[1], data[2], squared=False), 3))
    )

    # Discrete SIRD
    st.header("Discrete SIRD")

    col1, col2, col3 = st.beta_columns(3)

    region_selectbox = col1.selectbox(
        "Region:",
        dpc_regioni_df.denominazione_regione.unique(),
        int((dpc_regioni_df.denominazione_regione == 'Piemonte').argmax())
    )

    lags = col2.slider("Lags", 5, 15, 7)
    days_to_predict = col3.slider("Days to predict", 5, 30, 14)
    data_filter = '20200630'

    model = DeterministicSird(
        data_df=dpc_regioni_df,
        pop_prov_df=pop_prov_df,
        prov_list_df=prov_list_df,
        area=region_selectbox,
        group_column='denominazione_regione',
        data_column='data',
        data_filter=data_filter,
        lag=lags,
        days_to_predict=days_to_predict
    )

    res = model.fit()
    real_df = model.real_df

    st.plotly_chart(
        general_plot(
            t=real_df['data'],
            title='Infected of ' + region_selectbox,
            data=[
                real_df['nuovi_positivi'].values,
                res['nuovi_positivi'].values
            ],
            names=['Real', 'Prediction'],
            modes=['markers', 'lines'],
            blend_legend=False,
            output_image=False,
            output_figure=True
        ), use_container_width=True
    )

    st.plotly_chart(
        general_plot(
            t=real_df['data'],
            title='Cumulative deaths of ' + region_selectbox,
            data=[
                real_df['deceduti'].values,
                res['deceduti'].values
            ],
            names=['Real', 'Prediction'],
            modes=['markers', 'lines'],
            blend_legend=False,
            output_image=False,
            output_figure=True
        ), use_container_width=True
    )

    st.plotly_chart(
        general_plot(
            t=real_df['data'],
            title='Cumulative infected of ' + region_selectbox,
            data=[
                real_df['totale_positivi'].values,
                res['totale_positivi'].values
            ],
            names=['Real', 'Prediction'],
            modes=['markers', 'lines'],
            blend_legend=False,
            output_image=False,
            output_figure=True
        ), use_container_width=True
    )

    mae_tot_pos = model.mae(compart='totale_positivi')
    mse_tot_pos = model.mse(compart='totale_positivi')
    mae_deaths = model.mae(compart='deceduti')
    mse_deaths = model.mse(compart='deceduti')
    mae_rec = model.mae(compart='dimessi_guariti')
    mse_rec = model.mse(compart='dimessi_guariti')

    st.write(
        "Average MAE: " +
        str(np.round(np.mean([mae_tot_pos, mae_deaths, mae_rec]), 2))
    )
    st.write(
        "Average MSE: " +
        str(np.round(np.mean([mse_tot_pos, mse_deaths, mse_rec]), 2))
    )


def load_tf_page(covidpro_df, dpc_regioni_df):
    st.subheader("🚧 Page under construction")


def load_eda(covidpro_df, dpc_regioni_df):
    # Raw data checkbox
    show_raw_data = st.sidebar.checkbox('Show raw data')

    # --------------
    # Regional plots
    # --------------
    st.header("Regional plots")

    col1, col2, col3 = st.beta_columns(3)

    # Combobox
    region_selectbox = col1.selectbox(
        "Region:",
        dpc_regioni_df.denominazione_regione.unique(),
        int((dpc_regioni_df.denominazione_regione == 'Lombardia').argmax())
    )

    # Date pickers
    start_date_region = col2.date_input(
        'Start date',
        datetime.date(2020, 2, 24),
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]['data']
    )
    end_date_region = col3.date_input(
        'End date',
        dpc_regioni_df.iloc[-1]['data'],
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]['data']
    )

    # Filter data
    dpc_reg_filtered = dpc_regioni_df.query(
        end_date_region.strftime('%Y%m%d') +
        ' >= data >= ' +
        start_date_region.strftime('%Y%m%d')
    )

    if show_raw_data:
        dpc_reg_filtered[dpc_reg_filtered.denominazione_regione == region_selectbox]  # nopep8

    # Plots
    st.plotly_chart(
        custom_plot(
            df=dpc_reg_filtered,
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
            df=dpc_reg_filtered,
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

    # ----------------
    # Provincial plots
    # ----------------
    st.header("Provincial plots")

    col3, col4, col5 = st.beta_columns(3)

    # Combobox
    province_selectbox = col3.selectbox(
        "Province:",
        covidpro_df.Province.unique(),
        int((covidpro_df.Province == 'Piacenza').argmax())
    )

    # Date pickers
    start_date_province = col4.date_input(
        'Start date',
        datetime.date(2020, 2, 24),
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]['data'],
        'start_date_province'
    )
    end_date_province = col5.date_input(
        'End date',
        dpc_regioni_df.iloc[-1]['data'],
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]['data'],
        'end_date_province'
    )

    # Filter data
    covidpro_filtered = covidpro_df.query(
        end_date_province.strftime('%Y%m%d') +
        ' >= Date >= ' +
        start_date_province.strftime('%Y%m%d')
    )

    if show_raw_data:
        covidpro_filtered[covidpro_filtered.Province == province_selectbox]

    # Plots
    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
            xdata='Date',
            ydata=[
                'Deaths',
                'New_cases'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Individuals',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=[
                'Deceduti',
                'Nuovi casi'],
            template='simple_white'
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
            xdata='Date',
            ydata=[
                'Tot_deaths',
                'Curr_pos_cases'],
            title='COVID-19 - ',
            xtitle='Date',
            ytitle='Individuals',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=[
                'Totale deceduti',
                'Totale positivi'],
            template='simple_white'
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
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
