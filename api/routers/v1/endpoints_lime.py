import logging
# import os
import logging

from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
# from fastapi import Query

from setting.environment import set_env
from setting.logger import set_logger

router = APIRouter(default_response_class=JSONResponse)


@router.get(
    path='/ciao',
    summary=' ',
    description='',
    tags=['LIME']
)
async def ciao():

    set_env()
    set_logger(level='INFO')

    logging.info('LIME ME')

    return
