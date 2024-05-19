import pandas as pd
import os
import shap
import yaml

from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from fastapi import HTTPException
# from fastapi import Query
# from fastapi import status, BackgroundTasks

from setting.logger import set_logger
from core.xai.xai_shap import xai_global_shap
from core.xai.xai_shap import xai_local_shap


router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/build',
    summary=' ',
    description='',
    tags=['SHAP']
)
async def shap_build():

    set_logger(level='INFO')

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)['CHURN']
    str_path_viz = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    str_path_mod = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']

    os.environ['PATH_OUT_VIZ'] = str_path_viz
    os.environ['PATH_OUT_MOD'] = str_path_mod

    # load data
    try:
        model = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'model.pickle')
        # X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found at {str_path_viz}')

    # X_test = X_test.iloc[:10]

    # Use the SHAP library to explain the model's predictions
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)

    pd.to_pickle(shap_values, os.environ['PATH_OUT_MOD'] + 'shap_values.pickle')

    return {"message": "Task completed"}


@router.get(
    path='/global',
    summary=' ',
    description='',
    tags=['SHAP']
)
async def shap_global():

    set_logger(level='INFO')

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)['CHURN']
    str_path_viz = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    str_path_mod = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']

    os.environ['PATH_OUT_VIZ'] = str_path_viz
    os.environ['PATH_OUT_MOD'] = str_path_mod

    # load data
    try:
        shap_values = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'shap_values.pickle')
        # X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found at {str_path_viz}')

    os.makedirs(f'{os.environ['PATH_OUT_VIZ']}/SHAP', exist_ok=True)
    xai_global_shap(shap_values=shap_values, X_test=X_test, bln_save=True)

    return {"message": "Task completed"}


@router.get(
    path='/local',
    summary=' ',
    description='',
    tags=['SHAP']
)
async def shap_local(
        row: str
):

    set_logger(level='INFO')

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)['CHURN']
    str_path_viz = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    str_path_mod = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']

    os.environ['PATH_OUT_VIZ'] = str_path_viz
    os.environ['PATH_OUT_MOD'] = str_path_mod

    os.makedirs(f'{os.environ['PATH_OUT_VIZ']}/SHAP', exist_ok=True)

    # load data
    try:
        shap_values = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'shap_values.pickle')
        # X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found at {str_path_viz}')

    xai_local_shap(shap_value_cust=shap_values[row], cust_id=X_test.index.tolist()[row], bln_save=True)

    return {"message": "Task completed"}
