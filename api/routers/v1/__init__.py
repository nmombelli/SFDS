from fastapi.routing import APIRouter
from . import endpoints


def _build_router() -> APIRouter:
    rt = APIRouter(
        # tags=[{'A1': 'B1'}]
    )
    rt.include_router(endpoints.router, prefix="")

    return rt


router = _build_router()
