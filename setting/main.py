import logging

from api.exceptionhandlers import register_exception_handlers
from api.routers import register_routers
from fastapi import FastAPI
from fastapi.openapi.docs import (get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html)
from starlette.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


tags_metadata = [
    {'name': 'Model', 'description': None},
    {'name': 'SHAP', 'description': None},
    {'name': 'Internal', 'description': 'Reserved APIs'}
]

app = FastAPI(
    title='SFDS ️',
    description='Welcome! </br>'
                '</br>'
                'Feel free to Try Me, Venoso!',
    openapi_tags=tags_metadata,
    version='1.0',
    docs_url=None,
    redoc_url=None,
)

# https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs

try:
    app.mount("/static", StaticFiles(directory='static'), name="static")
except Exception as e:
    logging.warning(f"No static path - {e}")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="./openapi.json",
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        # swagger_js_url="./static/swagger-ui-bundle.js",
        # swagger_css_url="./static/swagger-ui.css",
        # swagger_favicon_url="./static/logo.png"
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect() -> HTMLResponse:
    return get_swagger_ui_oauth2_redirect_html()


@app.on_event("startup")
async def startup_event() -> None:
    pass


@app.on_event("shutdown")
def shutdown_event() -> None:
    pass


register_exception_handlers(app)
register_routers(app)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
