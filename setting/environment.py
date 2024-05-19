import os
import yaml


def set_env():

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)['CHURN']

    os.environ['PATH_OUT_VIZ'] = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    os.environ['PATH_OUT_MOD'] = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']

    return
