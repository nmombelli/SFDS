from fastapi.routing import APIRouter
from . import endpoints_model


def _build_router() -> APIRouter:
    rt = APIRouter(
        # tags=[{'A1': 'B1'}]
    )
    rt.include_router(endpoints_model.router, prefix="")

    return rt


router = _build_router()
