import os, sys  # noqa: E401
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.abspath('..'))
from src.utils import DataDownloader  # noqa: E402

if __name__ == "__main__":
    DataDownloader("../data").download_all_csv()
