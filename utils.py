import os
import wget
from pathlib import Path

class CSVUrl():
    COVIDPRO_CSV = "https://raw.githubusercontent.com/CEEDS-DEMM/COVID-Pro-Dataset/master/deathsItaProv.csv"
    DPC_REGIONI = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
    DPC_PROVINCE = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"

class CSVName():
    COVIDPRO_CSV = "deathsItaProv.csv"
    DPC_REGIONI = "dpc-covid19-ita-regioni.csv"
    DPC_PROVINCE = "dpc-covid19-ita-province.csv"
    POP_PROV_CSV = "pop_prov_age_3_groups_2020.csv"

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