import os
import wget
from pathlib import Path

import pandas as pd
import numpy as np


class CSVUrl():
    COVIDPRO_CSV = "https://raw.githubusercontent.com/CEEDS-DEMM/COVID-Pro-Dataset/master/deathsItaProv.csv"  # nopep8
    DPC_REGIONI = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"  # nopep8
    DPC_PROVINCE = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"  # nopep8


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
    """
    Computes the total population for a region
    starting from the provinces' popuplation

    Parameters
    ----------
    region : str
        The region whose population we need

    pop_df : pandas DataFrame
        Data for provinces population

    prov_df : pandas DataFrame
        Data that associates each province with
        its region

    Returns
    -------
    N : int
        The population of the region
    """

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

    dpc_province_df['data'] = pd.to_datetime(dpc_province_df['data']) \
        .dt.normalize()
    dpc_regioni_df['data'] = pd.to_datetime(dpc_regioni_df['data']) \
        .dt.normalize()
    covidpro_df['Date'] = pd.to_datetime(covidpro_df['Date']) \
        .dt.normalize()

    dpc_province_df.denominazione_provincia = \
        dpc_province_df.denominazione_provincia.str \
        .replace("Forlì-Cesena", "Forli-Cesena")

    dpc_regioni_df = fix_trentino(dpc_regioni_df)

    covidpro_df.Region = covidpro_df.Region.str \
        .replace("P.A. Trento", "Trentino Alto Adige") \
        .replace("P.A. Bolzano", "Trentino Alto Adige")

    covidpro_df.Province = covidpro_df.Province.str \
        .replace("Reggio nell Emilia", "Reggio nell'Emilia") \
        .replace("L Aquila", "L'Aquila") \

    pop_prov_df.Territorio = pop_prov_df.Territorio.str \
        .replace("Valle d'Aosta", "Aosta") \
        .replace("Forlì-Cesena", "Forli-Cesena") \
        .replace("Massa-Carrara", "Massa Carrara")

    # .replace("Reggio nell'Emilia", 'Reggio nell Emilia') \
    # .replace("L'Aquila", "L Aquila")

    covidpro_df.fillna(0, inplace=True)
    dpc_regioni_df.fillna(0, inplace=True)
    dpc_province_df.fillna(0, inplace=True)

    return (
        covidpro_df, dpc_regioni_df,
        dpc_province_df, pop_prov_df, prov_list_df)


def fix_trentino(df):
    """
    We need to sum all values by date for the
    two autonomous provinces of Trentino, and
    then replace these values with the original
    ones in the original df.
    """

    df_bolzano = df.loc[df.denominazione_regione == 'P.A. Bolzano']
    df_trento = df.loc[df.denominazione_regione == 'P.A. Trento']

    cols_to_drop = ['data', 'stato', 'codice_regione',
                    'denominazione_regione', 'lat', 'long',
                    'variazione_totale_positivi', 'note_test',
                    'note_casi', 'note']

    res_df = df_trento.drop(columns=cols_to_drop).reset_index(drop=True) + \
        df_bolzano.drop(columns=cols_to_drop).reset_index(drop=True)

    res_df.insert(0, 'data', df_bolzano.reset_index(drop=True)['data'])
    res_df.insert(0, 'denominazione_regione', 'Trentino Alto Adige')

    # We don't care about these atm
    res_df[[
        'stato',
        'codice_regione',
        'lat', 'long',
        'variazione_totale_positivi', 'note_test',
        'note_casi', 'note']] = 0

    df_filtered = df.loc[df.denominazione_regione != 'P.A. Bolzano']
    df_filtered = df_filtered.loc[
        df_filtered.denominazione_regione != 'P.A. Trento']

    df_res = pd.concat([df_filtered, res_df])
    df_res.sort_values(by=['data', 'denominazione_regione'], inplace=True)

    return df_res


def pre_process_csv(covidpro_df,
                    dpc_regioni_df,
                    dpc_province_df,
                    pop_prov_df,
                    prov_list_df,
                    window=7,
                    equalize_dates=True):

    dpc_regioni_df['NC_R'] = \
        dpc_regioni_df['nuovi_positivi']/dpc_regioni_df['tamponi']

    dpc_regioni_df['NP_R'] = \
        dpc_regioni_df['nuovi_positivi']/dpc_regioni_df['totale_positivi']

    dpc_regioni_df['IC_R'] = \
        dpc_regioni_df['terapia_intensiva']/dpc_regioni_df['totale_positivi']

    dpc_regioni_df['Hosp_R'] = \
        dpc_regioni_df['totale_ospedalizzati']/dpc_regioni_df['totale_positivi']  # nopep8

    dpc_regioni_df['DR'] = \
        dpc_regioni_df['deceduti']/dpc_regioni_df['totale_positivi']

    # ------------------------
    # Fixing outliers manually
    # ------------------------
    #
    # Fix Teramo & Chieti
    covidpro_df.loc[
        (covidpro_df.Province == 'Teramo') &
        (covidpro_df.Date == pd.Timestamp(2020, 6, 24)),
        'New_cases'] = 0
    covidpro_df.loc[
        (covidpro_df.Province == 'Teramo') &
        (covidpro_df.Date == pd.Timestamp(2020, 6, 24)),
        'Curr_pos_cases'] = 630
    covidpro_df.loc[
        (covidpro_df.Province == 'Teramo') &
        (covidpro_df.Date == pd.Timestamp(2020, 6, 25)),
        'New_cases'] = 1
    covidpro_df.loc[
        (covidpro_df.Province == 'Chieti') &
        (covidpro_df.Date == pd.Timestamp(2020, 6, 24)),
        'New_cases'] = 0
    covidpro_df.loc[
        (covidpro_df.Province == 'Chieti') &
        (covidpro_df.Date == pd.Timestamp(2020, 6, 24)),
        'Curr_pos_cases'] = 816
    covidpro_df.loc[
        (covidpro_df.Province == 'Chieti') &
        (covidpro_df.Date == pd.Timestamp(2020, 6, 25)),
        'New_cases'] = 0

    # Fix Bolzano
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 10, 7)),
        'New_cases'] = 55
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 10, 7)),
        'Curr_pos_cases'] = 3734
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 10, 8)),
        'New_cases'] = 69
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 26)),
        'New_cases'] = 24
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 26)),
        'Curr_pos_cases'] = 28746
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 26)),
        'Deaths'] = 14
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 26)),
        'Tot_deaths'] = 718
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 27)),
        'New_cases'] = 57
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 27)),
        'Tot_deaths'] = 720
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 28)),
        'Tot_deaths'] = 722
    covidpro_df.loc[
        (covidpro_df.Province == 'Bolzano') &
        (covidpro_df.Date == pd.Timestamp(2020, 12, 29)),
        'Tot_deaths'] = 729

    # Fix Rovigo
    covidpro_df.loc[
        (covidpro_df.Province == 'Rovigo') &
        (covidpro_df.Date == pd.Timestamp(2020, 4, 21)),
        'New_cases'] = 0

    covidpro_df.loc[
        (covidpro_df.Province == 'Rovigo') &
        (covidpro_df.Date == pd.Timestamp(2020, 6, 26)),
        'Deaths'] = 0

    # ----------

    covidpro_df['New_cases'] = covidpro_df['New_cases'].apply(convert_nan)
    covidpro_df['Deaths'] = covidpro_df['Deaths'].apply(convert_nan)

    covidpro_df['NP_R'] = covidpro_df.apply(compute_ratio_NP, axis=1)
    covidpro_df['DR'] = covidpro_df.apply(compute_ratio_DR, axis=1)

    dpc_regioni_df['NC_R_Rolling'] = dpc_regioni_df['NC_R'] \
        .rolling(window=window).mean()

    dpc_regioni_df['IC_R_Rolling'] = dpc_regioni_df['IC_R'] \
        .rolling(window=window).mean()

    dpc_regioni_df['totale_positivi_Rolling'] = \
        dpc_regioni_df['totale_positivi'].rolling(window=window).mean()

    covidpro_df['NP_R_Rolling'] = \
        covidpro_df['NP_R'].rolling(window=window).mean()

    covidpro_df['DR_Rolling'] = \
        covidpro_df['DR'].rolling(window=window).mean()

    if equalize_dates:
        prov_date = covidpro_df.iloc[-1]['Date']
        reg_date = dpc_regioni_df.iloc[-1]['data']

        if reg_date < prov_date:
            last_date = reg_date.strftime('%Y%m%d')
            covidpro_df = covidpro_df.query(f'{last_date} >= Date')
        else:
            last_date = prov_date.strftime('%Y%m%d')
            dpc_regioni_df = dpc_regioni_df.query(f'{last_date} >= data')

    covidpro_df.fillna(0, inplace=True)
    dpc_regioni_df.fillna(0, inplace=True)
    dpc_province_df.fillna(0, inplace=True)

    return (
        covidpro_df, dpc_regioni_df,
        dpc_province_df, pop_prov_df, prov_list_df)


def convert_nan(x):
    if x is np.NaN or x < 0:
        return 0
    else:
        return x


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
