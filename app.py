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
    st.subheader("🗺️ Data exploration")
    st.markdown("* This gives a general overview of the data with interactive "
                "plots.")
    st.subheader("📈 Time series")
    st.markdown("* This page allows you to see predictions made using time "
                "series models and the Prophet library.")
    st.subheader("👓 SIRD model")
    st.markdown("* This page allows you to see predictions made using "
                "stochastic and deterministic sird with time-dependent "
                "parameters.")
    st.subheader("🪄 TensorFlow model")
    st.markdown("* This page serves to show predictions made using "
                "neural networks (such as LSTM) implemented using "
                "TensorFlow.")


def load_ts_page(covidpro_df, dpc_regioni_df):
    st.subheader("🏗 Page under construction")


def load_tf_page(covidpro_df, dpc_regioni_df):
    st.subheader("🏗 Page under construction")


@st.cache
def compute_sird(prov, pop_prov_df, prov_list_df=None,
                 r0_start=3.5, r0_end=0.9, k=0.9,
                 x0=20, alpha=0.1, gamma=1/7):

    return sird(
        province=prov,
        pop_prov_df=pop_prov_df,
        prov_list_df=prov_list_df,
        gamma=gamma,
        alpha=alpha,
        R_0_start=r0_start,
        k=k,
        x0=x0,
        R_0_end=r0_end
    )


@st.cache
def data_sird_plot(covidpro_df,
                   column,
                   comp_array,
                   province_selectbox,
                   is_regional):

    return data_for_plot(
        compart='Infected',
        df=covidpro_df,
        column=column,
        comp_array=comp_array,
        province=province_selectbox,
        is_regional=is_regional
    )


def load_sird_page(covidpro_df, dpc_regioni_df, pop_prov_df, prov_list_df):
    # Page setup:
    # Sidebar widgets
    st.sidebar.header('Options')

    area_radio = st.sidebar.radio(
        "Regional or provincial predictions:",
        ['Regional', 'Provincial'],
        index=1
    )

    is_regional = True
    pcm_data = None
    group_column = 'denominazione_regione'
    data_df = dpc_regioni_df
    data_column = 'data'
    prov_df = prov_list_df
    column = "totale_positivi"

    if area_radio == "Regional":
        area_selectbox = st.sidebar.selectbox(
            "Region:",
            dpc_regioni_df.denominazione_regione.unique(),
            int((dpc_regioni_df.denominazione_regione == 'Piemonte').argmax()),
            key="area_selectbox_reg"
        )
    else:
        area_selectbox = st.sidebar.selectbox(
            "Province:",
            covidpro_df.Province.unique(),
            int((covidpro_df.Province == 'Firenze').argmax()),
            key="area_selectbox_prov"
        )
        is_regional = False
        pcm_data = dpc_regioni_df
        group_column = 'Province'
        data_df = covidpro_df
        data_column = 'Date'
        prov_df = None
        column = "New_cases"

    # ---------------
    # Continuous SIRD
    # ---------------
    st.header("Continuous SIRD")

    col1, col2, col3 = st.beta_columns(3)

    # Sird parameters
    r0_start = col1.slider("R0 start", 1.0, 6.0, 2.0)
    r0_end = col1.slider("R0 end", 0.01, 3.5, 0.3)
    k_value = col2.slider("R0 decrease rate", 0.01, 1.0, 0.2)
    x0_value = col2.slider("Lockdown day", 0, 100, 40)
    alpha_value = col3.slider("Death rate", 0.001, 1.0, 0.01)
    gamma_value = col3.slider("Recovery rate", 0.001, 1.0, 1/7)

    # Compute SIRD
    sirsol = compute_sird(area_selectbox, pop_prov_df, prov_df,
                          r0_start, r0_end, k_value,
                          x0_value, alpha_value, gamma_value)
    S, I, R, D = sirsol

    times = list(range(sirsol.shape[1]))

    # SIRD plot
    st.plotly_chart(
        general_plot(
            t=times,
            data=sirsol,
            title='SIRD model',
            traces_visibility=['legendonly'] + [True]*3,
            output_image=False,
            template='simple_white',
            output_figure=True,
            horiz_legend=True
        ), use_container_width=True
    )

    names, title, data, modes = data_sird_plot(
        data_df, column, I, area_selectbox, is_regional)

    # Comparison SIRD plot
    st.plotly_chart(
        general_plot(
            t=times,
            title='SIRD predictions comparison',
            data=data,
            names=names,
            modes=modes,
            blend_legend=False,
            output_image=False,
            traces_visibility=['legendonly'] + [True]*2,
            template='simple_white',
            output_figure=True,
            horiz_legend=True
        ), use_container_width=True
    )

    # Show metrics
    mae = mean_absolute_error(data[1], data[2])
    mse = mean_squared_error(data[1], data[2])
    rmse = mean_squared_error(data[1], data[2], squared=False)

    with st.beta_expander('Show metrics'):
        st.info("MAE: " + str(np.round(mae, 3)))
        st.info("MSE: " + str(np.round(mse, 3)))
        st.info(
            "RMSE: " +
            str(np.round(rmse, 3))
        )

    st.text("")
    st.text("")
    st.text("")
    # -------------
    # Discrete SIRD
    # -------------
    st.header("Discrete SIRD")

    col1_reg, col2_reg, = st.beta_columns(2)

    lags = col1_reg.slider("Lags", 5, 15, 7)
    days_to_predict = col2_reg.slider("Days to predict", 5, 30, 14)
    data_filter = '20200630'

    # Define and fit SIRD
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
        pcm_data=pcm_data
    )

    res = model.fit()
    real_df = model.real_df

    # Infected
    st.plotly_chart(
        general_plot(
            t=real_df['data'],
            title='Infected of ' + area_selectbox,
            data=[
                real_df['nuovi_positivi'].values,
                res['nuovi_positivi'].values
            ],
            names=['Real', 'Prediction'],
            modes=['markers', 'lines'],
            blend_legend=False,
            output_image=False,
            output_figure=True,
            xtitle='',
            horiz_legend=True
        ), use_container_width=True
    )

    # Deaths
    st.plotly_chart(
        general_plot(
            t=real_df['data'],
            title='Cumulative deaths of ' + area_selectbox,
            data=[
                real_df['deceduti'].values,
                res['deceduti'].values
            ],
            names=['Real', 'Prediction'],
            modes=['markers', 'lines'],
            blend_legend=False,
            output_image=False,
            output_figure=True,
            xtitle='',
            horiz_legend=True
        ), use_container_width=True
    )

    # Cumulative infected
    st.plotly_chart(
        general_plot(
            t=real_df['data'],
            title='Cumulative infected of ' + area_selectbox,
            data=[
                real_df['totale_positivi'].values,
                res['totale_positivi'].values
            ],
            names=['Real', 'Prediction'],
            modes=['markers', 'lines'],
            blend_legend=False,
            output_image=False,
            output_figure=True,
            xtitle='',
            horiz_legend=True
        ), use_container_width=True
    )

    # Show metrics
    mae_tot_pos = model.mae(compart='totale_positivi')
    mse_tot_pos = model.mse(compart='totale_positivi')
    mae_deaths = model.mae(compart='deceduti')
    mse_deaths = model.mse(compart='deceduti')
    mae_rec = model.mae(compart='dimessi_guariti')
    mse_rec = model.mse(compart='dimessi_guariti')

    with st.beta_expander('Show metrics'):
        st.info(
            "Average MAE: " +
            str(np.round(np.mean([mae_tot_pos, mae_deaths, mae_rec]), 2))
        )
        st.info(
            "Average MSE: " +
            str(np.round(np.mean([mse_tot_pos, mse_deaths, mse_rec]), 2))
        )

    st.text("")
    st.text("")
    st.subheader("🏗 Page under construction")


def load_eda(covidpro_df, dpc_regioni_df):
    st.sidebar.header('Options')

    # Date pickers
    start_date_eda = st.sidebar.date_input(
        'Start date',
        datetime.date(2020, 2, 24),
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]['data']
    )
    end_date_eda = st.sidebar.date_input(
        'End date',
        dpc_regioni_df.iloc[-1]['data'],
        datetime.date(2020, 2, 24),
        dpc_regioni_df.iloc[-1]['data']
    )

    # Raw data checkbox
    show_raw_data = st.sidebar.checkbox('Show raw data')

    # --------------
    # Regional plots
    # --------------
    st.header("Regional plots")

    # Combobox
    region_selectbox = st.selectbox(
        "Region:",
        dpc_regioni_df.denominazione_regione.unique(),
        int((dpc_regioni_df.denominazione_regione == 'Lombardia').argmax())
    )

    # Filter data
    dpc_reg_filtered = dpc_regioni_df.query(
        end_date_eda.strftime('%Y%m%d') +
        ' >= data >= ' +
        start_date_eda.strftime('%Y%m%d')
    )

    if show_raw_data:
        dpc_reg_filtered[dpc_reg_filtered.denominazione_regione == region_selectbox]  # nopep8

    # Plots
    st.plotly_chart(
        custom_plot(
            df=dpc_reg_filtered,
            ydata=[
                'totale_casi',
                'dimessi_guariti',
                'totale_positivi',
                'isolamento_domiciliare',
                'totale_ospedalizzati',
                'ricoverati_con_sintomi',
                'terapia_intensiva',
                'deceduti'],
            title='',
            xtitle='',
            ytitle='Individuals',
            group_column='denominazione_regione',
            area_name=region_selectbox,
            blend_legend=False,
            legend_titles=[
                'Total cases',
                'Total recovered',
                'Total positives',
                'Home quarantine',
                'Hospitalized',
                'Hospitalized with symptoms',
                'Intensive care',
                'Total deaths'],
            template='simple_white',
            show_title=False,
            horiz_legend=True
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=dpc_reg_filtered,
            ydata=[
                'IC_R', 'Hosp_R',
                'NC_R', 'NP_R'],
            title='',
            xtitle='',
            ytitle='Fraction',
            group_column='denominazione_regione',
            area_name=region_selectbox,
            blend_legend=True,
            legend_titles=[
                'IC over tot. cases',
                'Hospitalized over tot. cases',
                'Positives over tampons',
                'Positives over tot. positives'],
            template='simple_white',
            show_title=False,
            horiz_legend=True
        ), use_container_width=True)

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
        int((covidpro_df.Province == 'Piacenza').argmax())
    )

    # Filter data
    covidpro_filtered = covidpro_df.query(
        end_date_eda.strftime('%Y%m%d') +
        ' >= Date >= ' +
        start_date_eda.strftime('%Y%m%d')
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
            title='',
            xtitle='',
            ytitle='Individuals',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=[
                'Deaths',
                'New cases'],
            template='simple_white',
            show_title=False,
            horiz_legend=True
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
            xdata='Date',
            ydata=[
                'Tot_deaths',
                'Curr_pos_cases'],
            title='',
            xtitle='',
            ytitle='Individuals',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=False,
            legend_titles=[
                'Total deaths',
                'Total cases'],
            template='simple_white',
            show_title=False,
            horiz_legend=True
        ), use_container_width=True)

    st.plotly_chart(
        custom_plot(
            df=covidpro_filtered,
            xdata='Date',
            ydata=['NP_R', 'DR'],
            title='',
            xtitle='',
            ytitle='Fraction',
            group_column='Province',
            area_name=province_selectbox,
            blend_legend=True,
            legend_titles=[
                'Positives over total cases',
                'Deaths over total cases'
                ],
            template='simple_white',
            show_title=False,
            horiz_legend=True
        ), use_container_width=True)

    st.subheader("🏗 Page under construction")


if __name__ == "__main__":
    main()
