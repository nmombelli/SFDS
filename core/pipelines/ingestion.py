import os
import pandas as pd
import yaml


def ingestion(str_source: str) -> pd.DataFrame:
    """
    Load the dataset requested.
    :param str_source: label of the dataset name to load.
    :return: the desired dataset.
    """

    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)[str_source]

    str_path = dct_ing['PATH_MAIN']
    str_sub = dct_ing['INGESTION']['PATH_ING']
    str_name = dct_ing['INGESTION']['FILENAME']
    str_sep = dct_ing['INGESTION']['SEP']

    dtf_load = pd.read_csv(str_path + str_sub + str_name, sep=str_sep)

    return dtf_load
