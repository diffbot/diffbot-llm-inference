import time

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import ValidationError
from starlette.responses import JSONResponse

from server.rag_router import rag_router

from config import get_config

config = get_config()
server_url = config.get_server_url()

app = FastAPI(
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
    servers=[{"url": server_url}]
)

# handle CORS preflight requests
@app.options('/{rest_of_path:path}')
async def preflight_handler(request: Request, rest_of_path: str) -> Response:
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

# set CORS headers
@app.middleware("http")
async def add_CORS_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

app.include_router(rag_router)

from server.log import get_logstash_logger
logger = get_logstash_logger("diffbot_llm_api")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Pydantic validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=3333, reload=True)