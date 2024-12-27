from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from enum import Enum

class ResponseModel(BaseModel):
    status: int = 200
    message: str = None
    
class DiffbotAPIResponse(ResponseModel):
    url: str = ""
    type: str = ""
    title: str = ""
    data: Any = None
    dql_time: float = 0.0
    diffbotapi_time: float = 0.0
    webindex_time: float = 0.0
    
class QueryResponse(ResponseModel):
    dql_query: str = None
    hits: int = None
    data: Any = None
    articles: Any = None
    instruction: str = None
    page_url: str = None 
    
class ExtractionResponse(ResponseModel):
    data: Any = None

class SearchResponse(ResponseModel):
    query: list[str] = None
    search_results: Any = None
    instructions: str = None

class DQLRequest(BaseModel):
    size: int = 10
    type: str = "query"
    query: str = ""

class DQLResponse(ResponseModel):
    query: str = ""
    type: str = ""
    hits: int = 0
    data: Any = None
    page_url: str = None

class WebSearchRequest(BaseModel):
    text: str
    
class WebSearchResult(BaseModel):
    title: str = None
    url: str = None
    snippet: str = None

class DiffbotException(Exception):
    status_code: int = 200
    detail: str

class DiffbotAPIException(DiffbotException):
    page_url: str = None

class DQLException(DiffbotException):
    dql: str = None

 
class JSExecutionRequest(BaseModel):
    expression: str = None

class JSExecutionResponse(ResponseModel):
    expression: Any = None
    data: Any = None

class JSExecutionException(DiffbotException):
    expression: str = None
