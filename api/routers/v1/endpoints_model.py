# import logging
import os

from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Query

from core.routines.run_all import execute_main
from setting.environment import set_env
from setting.logger import set_logger

router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/training',
    summary=' ',
    description='Train the Churn Model. </br>'
                '</br>'
                '</br>'
                '`Parameters` </br>'
                '<ul>'
                '<li><b>split_strategy:</b> whether to use oversampling or not in the train/test split. </li>'
                '<li><b>bln_scale:</b> if True, data are scaled with normal scaling approach. </li>'
                '`Output` </br>'
                '</ul>'
                '</br>'
                'Return a .csv including, for each item of the test set, its features, the actual target and the '
                'predicted target.',
    tags=['Model']
)
async def api_model(
        split_strategy: str = Query(enum=['OVERSAMPLING', None]),
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

    set_env()
    set_logger(level='INFO')

    os.makedirs(os.environ['PATH_OUT_VIZ'], exist_ok=True)
    os.makedirs(os.environ['PATH_OUT_MOD'], exist_ok=True)

    execute_main(
        str_source='CHURN',
        split_strategy=split_strategy,
        bln_scale=bln_scale,
        random_state=random_state,
    )

    return {"message": "Task completed"}
