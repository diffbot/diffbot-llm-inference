from typing import Any, List, Union, AsyncIterable, Optional, AsyncIterator
import json
import re

from config import get_config
from llm.api_models import ChatCompletionRequestMessage, CompletionUsage, CreateChatCompletionRequest, \
    ChatCompletionResponse, ChatCompletionStreamResponse, ToolCall, ToolCalls, LLMException, Error, RequestGlobals
from llm.openai_gpt import OpenAIModel, get_openai_llm
from llm.llms import ModelID, Role
from llm.tool_call_llm import ToolCallLLM, FUNCTION_CALL_TOKEN, END_OF_TEXT_TOKEN, parse_tool_call
from server.log import get_logstash_logger

DIFFBOT_TOOL_USE_PROMPT = get_config().get_system_prompt()

DIFFBOT_ALTERNATIVE_PROMPT = """You are a helpful assistant without access to any functions. Use the information below to answer the users query.
"""

END_OF_SYSTEM_TOKEN = '======='
logger = get_logstash_logger("diffbot_tool_call_llm")
WHITESPACE = re.compile("\s+")


def contains_url(query):
    regex = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    url = re.findall(regex, query)
    return len(url) > 0


class DiffbotToolCallLLM(ToolCallLLM):
    supported_models = {ModelID.DIFFBOT_SMALL}

    def __init__(self):
        super().__init__()

        self.llm = get_openai_llm()

    @classmethod
    def is_supported(cls, model: ModelID) -> bool:
        return model and model in cls.supported_models

    def get_system_prompt(self, query: Optional[str] = None) -> str:
        return f'{DIFFBOT_TOOL_USE_PROMPT}\n{END_OF_SYSTEM_TOKEN}'


    def remove_system_prompt(self, message: str) -> str:
        idx = message.find(END_OF_SYSTEM_TOKEN)
        if idx == -1:
            return message
        return message[idx+len(END_OF_SYSTEM_TOKEN):]

    def chat_completion(self, request: CreateChatCompletionRequest, request_globals: RequestGlobals) \
            -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        return self.llm.chat_completion(request, request_globals)

    async def select_tool(
        self, 
        request: CreateChatCompletionRequest, 
        request_globals: RequestGlobals
    ) -> Any: 

        response = await self.llm.chat_completion(request, request_globals)
        if not response:
            return None
        if isinstance(response, AsyncIterable):
            return self.parse_response(response, request, request_globals)

        elif response.choices[-1]:
            tool_call_message = response.choices[-1].message.content
            if FUNCTION_CALL_TOKEN in tool_call_message.strip():
                tool_calls = parse_tool_call(tool_call_message)
                # check non-empty list of tool_calls
                if tool_calls and tool_calls.tool_calls: 
                    return tool_calls
            response.choices[-1].message.content = sanitize_response(response.choices[-1].message.content)

        return response

    def parse_content_delta(self, content_delta: str, buffer: List[str]) -> str:
        # next chunk should stop at whitespace
        ret = ""
        next_chunk = content_delta
        remaining_chunk = None

        match = WHITESPACE.search(content_delta)
        if match:
            idx = match.end()
            next_chunk = content_delta[:idx]
            remaining_chunk = content_delta[idx:]

        buffer.append(next_chunk)
        buffer_str = "".join(buffer)
        min_length = min(len(buffer_str), len(FUNCTION_CALL_TOKEN))
        if buffer_str[:min_length] != FUNCTION_CALL_TOKEN[:min_length]:
            buffer.clear()
            ret = buffer_str
        if remaining_chunk is not None:
            buffer.append(remaining_chunk)
        return ret

    async def parse_response(self, response, request: CreateChatCompletionRequest, request_globals: RequestGlobals):
        all_chunks = []  # all message.chunks being streamed by llm
        buffer = []  # latest chunks streamed by llm that cannot be yielded yet
        async for chunk in response:

            if chunk.choices and len(chunk.choices) == 1 and chunk.choices[0].delta.content:
                all_chunks.append(chunk.choices[0].delta.content)

            # if finished_reason is present, LLM stopped generating.
            # if buffer is not empty, it might contain function calls to be parsed.
            if chunk.choices and len(chunk.choices) == 1 and chunk.choices[0].finish_reason:
                buffer_str = "".join(buffer) + chunk.choices[0].delta.content

                if buffer_str.startswith(FUNCTION_CALL_TOKEN):
                    all_chunks_str = "".join(all_chunks)
                    yield parse_tool_call(all_chunks_str)
                else:
                    chunk.choices[0].delta.content = buffer_str
                    # send last tokens
                    reason = chunk.choices[0].finish_reason
                    chunk.choices[0].finish_reason = None
                    yield chunk
                    # send last chunk with empty content
                    chunk.choices[0].finish_reason = reason
                    chunk.choices[0].delta.content = ''
                    yield chunk
                continue

            if chunk.choices and len(chunk.choices) == 1 and chunk.choices[0].delta.content:
                new_content = self.parse_content_delta(chunk.choices[0].delta.content, buffer)
                chunk.choices[0].delta.content = new_content
                if len(new_content)>0:
                    yield chunk

async def combine_stream(non_stream_results, stream_result):
    for item in non_stream_results:
        yield item
    async for item in stream_result:
        yield item

async def sanitize_stream(stream_result):
    async for item in stream_result:
        delta_content = item.choices[-1].delta.content
        item.choices[-1].delta.content = sanitize_response(delta_content)
        yield item

def sanitize_response(text):
    if not text:
        return text
    text = text.replace(END_OF_TEXT_TOKEN, "")
    text = text.replace("</s>", "")
    return text

diffbot_tool_call_llm = None
def get_diffbot_tool_call_llm():
    global diffbot_tool_call_llm
    if not diffbot_tool_call_llm:
        diffbot_tool_call_llm = DiffbotToolCallLLM()
    return diffbot_tool_call_llm