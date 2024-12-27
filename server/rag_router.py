import json
import time

from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer

from pydantic import BaseModel, ValidationError
from models.utils import cleanNullValues
from server.log import get_logstash_logger
from services.kg_rag_service import KgRagService, get_kg_rag_service
from llm.api_models import ChatCompletionStreamResponse, CreateChatCompletionRequest, \
    ErrorResponse, LLMException, RequestGlobals
from llm.llms import ModelID

rag_router = APIRouter(
    prefix="/rag",
    tags=['rag']
)

logger = get_logstash_logger("diffbot_llm_api")

oauth2_shceme = OAuth2PasswordBearer(tokenUrl="token")

async def authentication(token: str = Depends(oauth2_shceme)):
   return token

# OpenAI api compatible: https://platform.openai.com/docs/api-reference/chat/create
@rag_router.post("/v1/chat/completions")
async def chat_completions(
    diffbot_token: str = Depends(authentication),
    request: CreateChatCompletionRequest = Body(...),
    kg_rag_service: KgRagService = Depends(get_kg_rag_service)
):

    try:
        request_globals = RequestGlobals()
        request_globals.diffbot_token = diffbot_token
        request_globals.timings["start_time"] = time.time()

        # filter out empty messages
        request.messages = [msg for msg in request.messages if msg.content]

        # support new text/image schema (kindof, we'll ignore images for now)
        for msg in request.messages:
            if type(msg.content) is list and 'text' in msg.content[0]:
                msg.content = msg.content[0]['text']

        llm_request = cleanNullValues(request.copy(deep=True).dict())
        llm_request = CreateChatCompletionRequest(**llm_request)

        result = await kg_rag_service.chat_completions(
            request=llm_request,
            request_globals=request_globals,
        )

        if request.stream:
            return StreamingResponse(
                stream_generator_with_usage_logging(result, diffbot_token, request_globals),
                media_type="text/event-stream"
            )
        else:
            return result
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except LLMException as e:
        logger.info(f"LLM exception: {e}", extra={"token": diffbot_token}, exc_info=True)
        return ErrorResponse(error=e.error)
    except Exception as e:
        logger.error(f"Error in chat completions. Exception: {e}", extra={"token": diffbot_token}, exc_info=True)
        return ErrorResponse(error=e)

async def stream_generator_with_usage_logging(result, diffbot_token, request_globals: RequestGlobals):
    async for item in result:
        if isinstance(item, BaseModel):
            yield f'data: {json.dumps(item.dict())}\n\n'

            if ("time_to_stream" not in request_globals.timings and isinstance(item, ChatCompletionStreamResponse)
                    and item.choices and len(item.choices) > 0 and item.choices[0].delta
                    and item.choices[0].delta.content and not item.choices[0].delta.content.isspace()):
                request_globals.timings["time_to_stream"] = (time.time() - request_globals.timings["start_time"]) * 1000

@rag_router.get("/v1/models")
def supported_models():
    data = [
        {
            "id": ModelID.DIFFBOT_SMALL,
            "object": "model",
            "owned_by": "diffbot"
        }
        ]
    body = {
        "object": "list",
        "data": data
    }
    return body

