import os
import wget
from pathlib import Path

import pandas as pd
import numpy as np


class CSVUrl():
    COVIDPRO_CSV = "https://raw.githubusercontent.com/CEEDS-DEMM/COVID-Pro-Dataset/master/deathsItaProv.csv"
    DPC_REGIONI = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
    DPC_PROVINCE = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"


class CSVName():
    COVIDPRO_CSV = "deathsItaProv.csv"
    DPC_REGIONI = "dpc-covid19-ita-regioni.csv"
    DPC_PROVINCE = "dpc-covid19-ita-province.csv"
    POP_PROV_CSV = "pop_prov_age_3_groups_2020.csv"
    PROV_LIST_CSV = "prov_list.csv"


class DataDownloader():
    def __init__(self, path="data", overwrite=True):
        self.path = path
        self.overwrite = overwrite

    def _download(self, url, filename):
        if self.overwrite and os.path.exists(filename):
            os.remove(filename)

        wget.download(url, filename)

    def download_all_csv(self):
        self.download_covidpro_csv()
        self.download_regioni_csv()
        self.download_province_csv()

    def download_covidpro_csv(self):
        self._download(CSVUrl.COVIDPRO_CSV,
                       str(Path(self.path, CSVName.COVIDPRO_CSV)))

    def download_regioni_csv(self):
        self._download(CSVUrl.DPC_REGIONI,
                       str(Path(self.path, CSVName.DPC_REGIONI)))

    def download_province_csv(self):
        self._download(CSVUrl.DPC_PROVINCE,
                       str(Path(self.path, CSVName.DPC_PROVINCE)))


def get_region_pop(region, pop_df, prov_df):
    prov_list = prov_df[prov_df.Region == region]['Province'].values

    N = 0
    for prov in prov_list:
        N += pop_df.loc[
            (pop_df.Territorio == prov) &
            (pop_df.Eta == "Total")
            ]['Value'].values[0]

    return N


def load_csv(path):
    covidpro_df = pd.read_csv(Path(path, CSVName.COVIDPRO_CSV))
    dpc_regioni_df = pd.read_csv(Path(path, CSVName.DPC_REGIONI))
    dpc_province_df = pd.read_csv(Path(path, CSVName.DPC_PROVINCE))
    pop_prov_df = pd.read_csv(Path(path, CSVName.POP_PROV_CSV))
    prov_list_df = pd.read_csv(Path(path, CSVName.PROV_LIST_CSV))

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

    covidpro_df.fillna(0, inplace=True)
    dpc_regioni_df.fillna(0, inplace=True)
    dpc_province_df.fillna(0, inplace=True)

    return covidpro_df, dpc_regioni_df, dpc_province_df, pop_prov_df, prov_list_df


def pre_process_csv(covidpro_df,
                    dpc_regioni_df,
                    dpc_province_df,
                    pop_prov_df,
                    prov_list_df,
                    window=7):

    dpc_regioni_df['NC_R'] = \
        dpc_regioni_df['nuovi_positivi']/dpc_regioni_df['tamponi']

    dpc_regioni_df['NP_R'] = \
        dpc_regioni_df['nuovi_positivi']/dpc_regioni_df['totale_positivi']

    dpc_regioni_df['IC_R'] = \
        dpc_regioni_df['terapia_intensiva']/dpc_regioni_df['totale_positivi']

    dpc_regioni_df['Hosp_R'] = \
        dpc_regioni_df['totale_ospedalizzati']/dpc_regioni_df['totale_positivi']

    dpc_regioni_df['DR'] = \
        dpc_regioni_df['deceduti']/dpc_regioni_df['totale_positivi']

    lmb_rep = lambda x: 0 if x is np.NaN or x < 0 else x
    covidpro_df['New_cases'] = covidpro_df['New_cases'].apply(lmb_rep)
    covidpro_df['Deaths'] = covidpro_df['Deaths'].apply(lmb_rep)

    covidpro_df['NP_R'] = covidpro_df.apply(compute_ratio_NP, axis=1)
    covidpro_df['DR'] = covidpro_df.apply(compute_ratio_DR, axis=1)

    dpc_regioni_df['NC_R_Rolling'] = dpc_regioni_df['NC_R'] \
        .rolling(window=window).mean()

    dpc_regioni_df['IC_R_Rolling'] = dpc_regioni_df['IC_R'] \
        .rolling(window=window).mean()

    dpc_regioni_df['totale_positivi_Rolling'] = \
        dpc_regioni_df['totale_positivi'].rolling(window=window).mean()

    covidpro_df['NP_R_Rolling'] = covidpro_df['NP_R'].rolling(window=window).mean()
    covidpro_df['DR_Rolling'] = covidpro_df['DR'].rolling(window=window).mean()

    covidpro_df.fillna(0, inplace=True)
    dpc_regioni_df.fillna(0, inplace=True)
    dpc_province_df.fillna(0, inplace=True)

    return covidpro_df, dpc_regioni_df, dpc_province_df, pop_prov_df, prov_list_df


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


def load_data(path):
    return pre_process_csv(*load_csv(path))
