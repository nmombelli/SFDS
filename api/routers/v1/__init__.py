from fastapi.routing import APIRouter
from . import endpoints_model
from . import endpoints_shap
from . import endpoints_lime
from . import endpoints_safe


def _build_router() -> APIRouter:
    rt = APIRouter(
        # tags=[{'A1': 'B1'}]
    )
    rt.include_router(endpoints_model.router, prefix="/model")
    rt.include_router(endpoints_lime.router, prefix="/LIME")
    rt.include_router(endpoints_shap.router, prefix="/SHAP")
    rt.include_router(endpoints_safe.router, prefix="/SAFE")
    return rt


router = _build_router()
