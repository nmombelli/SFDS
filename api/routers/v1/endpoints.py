import logging
import json
import pandas as pd
import os
import yaml

from setting.logger import set_logger
from core.routines.run_all import execute_main
from core.routines.run_viz import run_eicer_viz

from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from fastapi import HTTPException, Query
# from fastapi import status, BackgroundTasks


router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/generate_ranking',
    summary=' ',
    description='Generate the ranking for the experiment items. </br>'
                '</br>'
                '</br>'
                '`Parameters` </br>'
                '<ul>'
                '<li><b>str_source:</b> name of the experiment to consider. Allowed values are HR and CHURN. </li>' 
                '<li><b>split_strategy:</b> whether to use oversampling or not in the train/test split. </li>'
                '<li><b>n_splits:</b> number of folds used for cross validation. </li>'
                '<li><b>bln_scale:</b> if True, data are scaled with normal scaling approach. </li>'
                '<li><b>logit_type:</b> how to perform feature selection. Allowed values are: </li>'
                '<ul>'
                '<li>None: no particular strategy applied; the model uses all the features received. </li>'
                '<li>BACK: backward stepwise procedure applied to select only the relevant features. </li>'
                '<li>FORWARD: forward stepwise procedure applied to select only the relevant features. </li>'
                '</ul>'
                '<li><b>log_level</b>: detail level of the log. </li>'
                '<li><b>random_state</b>: seed to be set for reproducibility. </li>'
                '</ul>'
                '</br>'
                '`Output` </br>'
                '</br>'
                'Return a .csv including, for each item, the values of the main features exploited by the model, '
                'the predictive score associated and the resulting ranking position. </br>'
    ,
    tags=['Ranking']
)
async def api_model(
        str_source: str = Query(enum=['HR', 'CHURN']),
        split_strategy: str = Query(default=None, enum=['OVERSAMPLING']),
        n_splits: int = Query(enum=[3, 5, 10]),
        logit_type: str = Query(default=None, enum=['BACK', 'FORWARD']),
        log_level: str = Query(enum=['DEBUG', 'INFO', 'WARNING', 'ERROR']),
        random_state: int = None
):

    set_logger(level=log_level)

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)[str_source]
    str_path_viz = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    str_path_mod = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']
    os.environ['PATH_OUT_VIZ'] = str_path_viz
    os.environ['PATH_OUT_MOD'] = str_path_mod

    os.makedirs(str_path_viz, exist_ok=True)
    os.makedirs(str_path_mod, exist_ok=True)

    # loading data, preparing data and running the logistic model
    execute_main(
        str_source=str_source,
        split_strategy=split_strategy,
        n_splits=n_splits,
        logit_type=logit_type,
        random_state=random_state,
    )

    return {"message": "Task completed"}


@router.get(
    path='/item_contrastive_xai',
    summary=' ',
    description='In the context of the selected experiment, it returns the Evaluative Item-Contrastive Explanations for'
                ' the ranking of two designated items. </br>'
                '</br>'
                '`Parameters` </br>'
                '<ul>'
                '<li><b>str_source:</b> name of the experiment to consider. Allowed values are HR and CHURN. </li>'
                '<li><b>rank_high:</b> position of the first item to compare. </li>'
                '<li><b>rank_low:</b> position of the second item to compare (must be lower than rank_high). </li>'
                '<li><b>scale:</b> scaling applied to the contributions values for the plotting phase. </li>'
                '<li><b>level_detail:</b> the amount of information explaining the ranking difference that the user '
                'wants to visualize. Allowed values are LOW, MEDIUM, HIGH, ALL, each corresponding to a certain '
                'percentage. </li>'
                '</ul>'
                '</br>'
                '`Output` </br>'
                '</br>'
                'Return the plot illustrating the ranking comparison between the two items as .png file.'
    ,
    tags=['Evaluative']
)
async def api_eicer(
        str_source: str = Query(enum=['HR', 'CHURN']),
        rank_high: int = None,
        rank_low: int = None,
        scale: str = Query(None, enum=['log', 'prc']),
        level_detail: str = Query(enum=['ALL', 'HIGH', 'MEDIUM', 'LOW']),
):

    set_logger(level='INFO')

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)[str_source]
    str_path_viz = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    str_path_mod = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']

    os.environ['PATH_OUT_VIZ'] = str_path_viz
    os.environ['PATH_OUT_MOD'] = str_path_mod

    # load data
    try:
        dtf_exp = pd.read_csv(str_path_viz + f'/RANKING_{str_source}.csv', sep=';')
    except Exception:
        raise HTTPException(status_code=404, detail=f'File RANKING_{str_source}.csv not found at {str_path_viz}')
    try:
        with open(str_path_mod + f"COEF_LOGIT_{str_source}.json", 'r') as f:
            dct_coef = json.load(f)
    except Exception:
        raise HTTPException(status_code=404, detail=f'File COEF_LOGIT_{str_source}.json not found at {str_path_mod}')

    str_idx = 'ID' if str_source == 'HR' else 'CLIENTNUM'
    dtf_exp.set_index(str_idx, inplace=True)

    idx_h = dtf_exp[dtf_exp['RANK'] == rank_high].index[0]
    idx_l = dtf_exp[dtf_exp['RANK'] == rank_low].index[0]

    logging.info(f'Selected items are: item={idx_h} with rank {rank_high} and item={idx_l} with rank {rank_low}')

    run_eicer_viz(
        dct_coef=dct_coef,
        dtf_exp=dtf_exp,
        scale=scale,
        idx_h=idx_h,
        idx_l=idx_l,
        level_detail=level_detail,
    )
    logging.info(f'PLOT stored in {str_path_viz}')

    return {"message": "Task completed"}
