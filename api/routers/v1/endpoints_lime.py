import dill
import lime
import lime.lime_tabular
import pandas as pd
import os

from fastapi import HTTPException
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
# from fastapi import Query

from setting.environment import set_env
from setting.logger import set_logger
from core.xai.xai_lime import xai_local_lime, xai_global_lime

router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/build',
    summary=' ',
    description='',
    tags=['LIME']
)
async def lime_build():

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        # X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found')

    # LIME has one explainer for all the models
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test.values,
        feature_names=X_test.columns.values.tolist(),
        class_names=['STAY', 'CHURN'],
        verbose=False,
        mode='classification'
    )

    os.makedirs(os.environ['PATH_OUT_LIME'], exist_ok=True)
    with open(os.environ['PATH_OUT_LIME'] + 'lime_explainer.pickle', 'wb') as f:
        dill.dump(lime_explainer, f)

    return {"message": "Task completed"}


@router.get(
    path='/global',
    summary=' ',
    description='',
    tags=['LIME']
)
async def lime_global():

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        with open(os.environ['PATH_OUT_LIME'] + 'lime_explainer.pickle', 'rb') as b:
            lime_explainer = dill.load(b)
        model = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'model.pickle')
        # X_train = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_train.pickle')
        X_test = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'X_test.pickle')
    except Exception:
        raise HTTPException(status_code=404, detail=f'Files not found')

    xai_global_lime(lime_explainer=lime_explainer, X_test=X_test, model=model)

    return {"message": "Task completed"}


@router.get(
    path='/local',
    summary=' ',
    description='',
    tags=['LIME']
)
async def lime_local(
    cust_id: int
):

    set_env()
    set_logger(level='INFO')

    # load data
    try:
        with open(os.environ['PATH_OUT_LIME'] + 'lime_explainer.pickle', 'rb') as b:
            lime_explainer = dill.load(b)
        model = pd.read_pickle(os.environ['PATH_OUT_MOD'] + 'model.pickle')
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

    xai_local_lime(
        lime_explainer=lime_explainer,
        data_row=X_test.values[cust_pos],
        model=model,
        cust_id=cust_id,
    )

    return {"message": "Task completed"}
