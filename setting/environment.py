import os
import yaml


def set_env():

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)['CHURN']

    os.environ['PATH_OUT_MOD'] = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']
    os.environ['PATH_OUT_LIME'] = dct_ing['PATH_MAIN'] + dct_ing['LIME']['PATH_LIME']
    os.environ['PATH_OUT_SHAP'] = dct_ing['PATH_MAIN'] + dct_ing['SHAP']['PATH_SHAP']
    os.environ['PATH_OUT_SAFE'] = dct_ing['PATH_MAIN'] + dct_ing['SAFE']['PATH_SAFE']

    return
