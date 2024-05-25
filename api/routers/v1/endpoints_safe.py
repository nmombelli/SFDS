import json
import pandas as pd
import os

from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from fastapi import HTTPException
# from fastapi import Query
# from fastapi import status, BackgroundTasks

from core.xai.xai_safe import xai_safe_global, xai_safe_local
from setting.environment import set_env
from setting.logger import set_logger


router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/global',
    summary=' ',
    description='',
    tags=['SAFE']
)
async def safe_global():

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        with open(os.environ['PATH_OUT_MOD'] + 'model_params.json', 'r') as fp:
            model_params = json.load(fp)
        X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
        y_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'y_train.pickle')
        y_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'y_test.pickle')

    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found')

    xai_safe_global(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_params=model_params,
    )

    return {"message": "Task completed"}


@router.get(
    path='/local',
    summary=' ',
    description='',
    tags=['SAFE']
)
async def safe_local(
    cust_id: int
):

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        with open(os.environ['PATH_OUT_MOD'] + 'model_params.json', 'r') as fp:
            model_params = json.load(fp)
        X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
        y_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'y_train.pickle')
        y_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'y_test.pickle')
        test_set = pd.read_csv(os.environ['PATH_OUT_MOD'] + 'test_set.csv', sep=';')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found')

    if cust_id not in X_test.index:
        dct_adv = {
            'MESSAGE': f'Customer with ID={cust_id} not in test dataset. You can try one of the following ones:',
            'CHURNING': test_set.loc[test_set['MODEL_PRED'] == 1, 'CLIENTNUM'].tolist()[:5],
            'NOT CHURNING': test_set.loc[test_set['MODEL_PRED'] == 0, 'CLIENTNUM'].tolist()[:5],
        }
        raise HTTPException(status_code=404, detail=dct_adv)

    xai_safe_local(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_params=model_params,
        cust_id=cust_id,
    )

    return {"message": "Task completed"}
