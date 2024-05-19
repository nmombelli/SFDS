# import logging
import os
import yaml

from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Query

from core.routines.run_all import execute_main
from setting.logger import set_logger

router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/model/training',
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
                'the predictive score associated and the resulting ranking position.</br>',
    tags=['Model']
)
async def api_model(
        split_strategy: str = Query(default=None, enum=['OVERSAMPLING']),
        bln_scale: bool = Query(enum=[False, True]),
        random_state: int = None
):
    """
    Loading data, preparing data and running the classification model
    :param split_strategy: how to split the dataset in train and test set. Allowed values are None and OVERSAMPLING.
    :param bln_scale: if True, train and test set are scaled with normal scaling approach.
    :param random_state: seed to be set for reproducibility
    :return: dictionary with the model parameters and the test dataframe with its predictions.
    """

    set_logger(level='INFO')

    # configuring the path to store the outputs
    with open(os.path.abspath('config/ingestion.yaml'), 'r') as f:
        dct_ing = yaml.safe_load(f)['CHURN']
    str_path_viz = dct_ing['PATH_MAIN'] + dct_ing['VIZ']['PATH_VIZ']
    str_path_mod = dct_ing['PATH_MAIN'] + dct_ing['MODEL']['PATH_MODEL']
    os.environ['PATH_OUT_VIZ'] = str_path_viz
    os.environ['PATH_OUT_MOD'] = str_path_mod

    os.makedirs(str_path_viz, exist_ok=True)
    os.makedirs(str_path_mod, exist_ok=True)

    execute_main(
        str_source='CHURN',
        split_strategy=split_strategy,
        bln_scale=bln_scale,
        random_state=random_state,
    )

    return {"message": "Task completed"}
