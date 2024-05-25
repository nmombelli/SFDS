import numpy as np
import pandas as pd
import os
import shap

from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from fastapi import HTTPException
# from fastapi import Query
# from fastapi import status, BackgroundTasks

from core.xai.xai_shap import xai_shap_global
from core.xai.xai_shap import xai_shap_local
from setting.environment import set_env
from setting.logger import set_logger


router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/build',
    summary=' ',
    description='',
    tags=['SHAP']
)
async def shap_build():

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        model = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'model.pickle')
        X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found')

    # Use the SHAP library to explain the model's predictions
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test)

    os.makedirs(os.environ['PATH_OUT_SHAP'], exist_ok=True)
    pd.to_pickle(shap_values, os.environ['PATH_OUT_SHAP'] + 'shap_values.pickle')

    return {"message": "Task completed"}


@router.get(
    path='/global',
    summary=' ',
    description='',
    tags=['SHAP']
)
async def shap_global():

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        shap_values = pd.read_pickle(os.environ['PATH_OUT_SHAP'] + 'shap_values.pickle')
        # X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found')

    xai_shap_global(shap_values=shap_values, X_test=X_test)

    return {"message": "Task completed"}


@router.get(
    path='/local',
    summary=' ',
    description='',
    tags=['SHAP']
)
async def shap_local(
        cust_id: int
):

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        shap_values = pd.read_pickle(os.environ['PATH_OUT_SHAP'] + 'shap_values.pickle')
        # X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
        test_set = pd.read_csv(os.environ['PATH_OUT_MOD'] + 'test_set.csv', sep=';')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found')

    try:
        cust_pos = X_test.index.get_loc(cust_id)
    except Exception:
        dct_adv = {
            'MESSAGE': f'Customer with ID={cust_id} not in test dataset. You can try one of the following ones:',
            'CHURNING': test_set.loc[test_set['MODEL_PRED'] == 1, 'CLIENTNUM'].tolist()[:5],
            'NOT CHURNING': test_set.loc[test_set['MODEL_PRED'] == 0, 'CLIENTNUM'].tolist()[:5],
        }
        raise HTTPException(status_code=404, detail=dct_adv)

    if not np.array_equal(np.array(X_test.loc[cust_id]), shap_values[cust_pos].data):
        raise HTTPException(status_code=404, detail=f'Data Not Matching')

    xai_shap_local(shap_value_cust=shap_values[cust_pos], cust_id=cust_id)

    return {"message": "Task completed"}
