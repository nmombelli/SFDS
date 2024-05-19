from fastapi.applications import FastAPI

from . import v1


def register_routers(app: FastAPI) -> FastAPI:
    app.include_router(v1.router, prefix="")
    # app.include_router(v1.router, prefix="shap")
    # app.include_router(v1.router, prefix="lime")
    return app

