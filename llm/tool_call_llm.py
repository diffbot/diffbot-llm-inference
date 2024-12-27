import json
import time
from abc import abstractmethod
from typing import Any, List, Optional, Dict, Union, AsyncIterable
import asyncio

from llm.api_models import (ChatCompletionRequestMessage, CompletionUsage, CreateChatCompletionRequest,
                            ChatCompletionResponse, ChatCompletionStreamResponse, Error, LLMException, ToolCall,
                            ToolCalls, RequestGlobals)
from llm.llms import LLM, Role, ModelID
from models.api import ResponseModel, DiffbotAPIResponse, DQLResponse
from server.log import get_logstash_logger
from llm.plugin import get_plugin, Plugin

logger = get_logstash_logger("tool_call_llm")

MAX_NUM_CALLS = 3
FUNCTION_CALL_TOKEN = '<functioncall>'
END_OF_TEXT_TOKEN = '<|endoftext|>'

def parse_tool_call(content) -> ToolCalls:
    try:
        content = content.lstrip()
        function_calls_str = content.split(FUNCTION_CALL_TOKEN)
        ret = []
        for call_str in function_calls_str:
            call_str = call_str.strip()
            # we don't expect the function call JSON to have linebreaks. The LLM sometimes adds additional paragraphs of
            # text after the function call and we want to ignore these additional paragraphs.
            if "\n" in call_str:
                call_str = call_str[:call_str.index("\n")]
            if not call_str:
                continue
            start_idx = 0
            end_idx = call_str.index(END_OF_TEXT_TOKEN) if END_OF_TEXT_TOKEN in call_str else len(call_str)
            json_text = call_str[start_idx:end_idx].strip()
            if not json_text.startswith("{") or not json_text.endswith("}"):
                continue
            try:
                function_call = json.loads(json_text)
                function_name = function_call['name']
                function_arguments = json.dumps(function_call['arguments'])
                ret.append(
                    ToolCall(function_name=function_name, function_arguments=function_arguments, tool_call_id=""))
            except Exception as e:
                logger.error(f"Failed to parse tool call: {json_text}. Exception: {e}", exc_info=True)
        if not ret:
            raise LLMException(error=Error(code=500, message="Invalid tool call. {}".format(content)))
        return ToolCalls(tool_calls=ret, content=content)
    except Exception:
        raise LLMException(error=Error(code=500, message="Invalid tool call. {}".format(content)))

async def combine_streams(stream_result1: AsyncIterable, stream_result2: AsyncIterable):
    async for item in stream_result1:
        yield item
    async for item in stream_result2:
        yield item

async def async_iterable(data):
    for item in data:
        yield item

class ToolCallLLM(LLM):

    @abstractmethod
    def get_system_prompt(self, query: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def remove_system_prompt(self, message: str) -> str:
        pass

    @abstractmethod
    def select_tool(
        self,
        request: CreateChatCompletionRequest,
        request_globals: RequestGlobals,
    ) -> Any:
        pass

    async def process_tool_calls(
        self,
        request: CreateChatCompletionRequest,
        plugin: Plugin,
        request_globals: RequestGlobals
    ) -> Any:
        # setup the request for tool calls
        tool_call_request = request.copy(deep=True)

        # add/update system prompt for tool calls
        start_system_prompt = time.time()
        system_prompt = self.get_system_prompt(query=tool_call_request.messages[-1].content)
        has_system = False
        if tool_call_request.messages and len(tool_call_request.messages) > 0:
            for message in tool_call_request.messages:
                if message.role == 'system':
                    message.content = f'{system_prompt}\n\n{message.content}'
                    has_system = True
                    break
        if not has_system:
            tool_call_request.messages.insert(0, ChatCompletionRequestMessage(role='system', content=system_prompt))
        request_globals.timings["system_prompt"] = (time.time() - start_system_prompt) * 1000

        request_globals.diffbot_responses = []
        if request.stream:
            return self._process_stream_tool_calls(tool_call_request, request_globals, plugin)
        return await self._process_non_stream_tool_calls(tool_call_request, request_globals, plugin)

    async def _process_non_stream_tool_calls(
            self,
            request: CreateChatCompletionRequest,
            request_globals: RequestGlobals,
            plugin: Plugin,
            num_tool_calls=0,
            force_last_tool_call=False
    ):
        response = await self.select_tool_from_llm_or_user(request, request_globals)
        if isinstance(response, ToolCalls) and len(response.tool_calls)>0 and isinstance(response.tool_calls[0], ToolCall):
            num_tool_calls += 1

            if force_last_tool_call:
                raise LLMException(error=Error(code=500, message=f"Exceeded max function call attempts"))

            if num_tool_calls > MAX_NUM_CALLS:
                # try to generate a response without tool calls as a last attempt
                await self.change_request_to_skip_function_calling(request)
                force_last_tool_call = True
            else:
                allow_fallback = not request_globals.disable_tool_fallback
                await self.invoke_function_call(request, request_globals, plugin, response,
                                                allow_fallback=allow_fallback)
            return await self._process_non_stream_tool_calls(request, request_globals, plugin,
                                                             num_tool_calls=num_tool_calls,
                                                             force_last_tool_call=force_last_tool_call)
        elif isinstance(response, ChatCompletionResponse):
            request.messages.append(response.choices[0].message)
            return response

        raise LLMException(error=Error(code=500, message="Invalid function call"))

    async def _process_stream_tool_calls(
            self,
            request: CreateChatCompletionRequest,
            request_globals: RequestGlobals,
            plugin: Plugin,
            num_tool_calls=0,
            force_last_tool_call=False
    ):
        response = await self.select_tool_from_llm_or_user(request, request_globals)
        if isinstance(response, ToolCalls):
            # emulate async generation of tool call when tool call is provided by user
            response = async_iterable([response])

        async for item in response:
            if not isinstance(item, ToolCalls):
                yield item
                continue
            # call the tool, make a completion request, and yield the LLM response here
            num_tool_calls += 1
            if force_last_tool_call:
                raise LLMException(error=Error(code=500, message=f"Exceeded max function call attempts"))

            if num_tool_calls > MAX_NUM_CALLS:
                # try to generate a response without tool calls as a last attempt
                await self.change_request_to_skip_function_calling(request)
                force_last_tool_call = True
            else:
                # now process the tool call
                allow_fallback = not request_globals.disable_tool_fallback
                await self.invoke_function_call(request, request_globals, plugin, item, allow_fallback=allow_fallback)
            ret = self._process_stream_tool_calls(request, request_globals, plugin,
                                                  num_tool_calls=num_tool_calls,
                                                  force_last_tool_call=force_last_tool_call)
            async for ret_item in ret:
                yield ret_item


    async def select_tool_from_llm_or_user(self, request, request_globals):
        # if user requests particular tool request, call function directly without LLM interaction.
        if request.messages[-1].content.startswith(FUNCTION_CALL_TOKEN):
            last_message = request.messages[-1].content
            # tool call request and response are added again after invocation
            request.messages.pop()
            response = parse_tool_call(last_message)
        else:
            response = await self.select_tool(request, request_globals)
        return response

    async def invoke_function_call(self, request: CreateChatCompletionRequest,
                                   request_globals: RequestGlobals,
                                   plugin,
                                   tool_calls: ToolCalls,
                                   allow_fallback: bool = False):
        invoked_tool_calls = []
        call_tasks = []
        tool_call_content = ''
        for tool_call in tool_calls.tool_calls:
            invoke_function_name = tool_call.function_name
            if not tool_call.function_name or not tool_call.function_arguments:
                logger.error("Invalid function call: " + str(tool_calls))
                continue
            tool_call_content += "<functioncall>" + json.dumps({"name": tool_call.function_name, "arguments": json.loads(tool_call.function_arguments)}) + '\n'
            call_tasks.append(plugin.invoke(
                function_name = invoke_function_name,
                function_arguments = json.loads(tool_call.function_arguments),
                token = request_globals.diffbot_token)
            )
            invoked_tool_calls.append(tool_call)
        plugin_responses = await asyncio.gather(*call_tasks)
        if tool_calls.content:
            request.messages.append(ChatCompletionRequestMessage(
                role=Role.assistant,
                content=tool_calls.content
            ))
        elif tool_call_content:
            request.messages.append(ChatCompletionRequestMessage(
                role=Role.assistant,
                content=tool_call_content.strip()
            ))

        request_with_call = request.copy(deep=True)

        request.messages.append(ChatCompletionRequestMessage(
            role=Role.user,
            tool_call_id="",
            content=""
        ))

        tool_responses = []
        for index, plugin_response in enumerate(plugin_responses):
            tool_call = invoked_tool_calls[index]
            if plugin_response is None:
                continue
            plugin_content = plugin_response.dict().get('content', {})
            tool_responses.append(plugin_content)
            request.messages[-1].tool_call_id = tool_call.tool_call_id  # use last tool_call_id
            request.messages[-1].tool_call_id = request.messages[-1].tool_call_id.strip()

            request_globals.diffbot_responses.append({
                "diffbot_request": {
                    "name": tool_call.function_name,
                    "arguments": json.loads(tool_call.function_arguments),
                },
                "diffbot_response": plugin_content
            })

        # tool message should be a valid json so that it can be truncated as a json later
        if tool_responses:
            request.messages[-1].content = json.dumps(tool_responses)

        # now we'll check if results are good enough. if they are not, fallback to web_search.

        if not allow_fallback:
            return

        tool_call_names = [tool_call.function_name for tool_call in invoked_tool_calls]
        if "web_search_v1" in tool_call_names or "dql_v1" not in tool_call_names:
            # fallback is for dql tool calls only for now
            return

        results_are_good = await self.verify_results(invoked_tool_calls, plugin_responses)
        if results_are_good:
            return

        start_fallback = time.time()
        fallback_request = request_with_call
        original_calls = request_with_call.messages[-1].content
        fallback_request.messages.append(
            ChatCompletionRequestMessage(
                content="""This function call does not return satisfactory results. Try again, now using web_search_v1. Your answer MUST start with: <functioncall> {"name": "web_search_v1", ...""",
                role="user")
        )
        fallback_request.stream=False
        # Fallback request should have the same prefix as original request to leverage vLLM's prefix caching.
        response = await self.select_tool(fallback_request, request_globals)
        logger.info(f"fallback function call request completed",
                    extra={
                        "fallback_request_time": round((time.time() - start_fallback) * 1000),
                        "llm_response": str(response),
                        "original_calls": original_calls,
                        "is_tool_call_response": isinstance(response, ToolCalls)
                    })

        if isinstance(response, ToolCalls) and response.tool_calls and response.tool_calls[0].function_name == "web_search_v1":
            # remove last dql response
            request_globals.diffbot_responses.pop()
            # remove last tool call response
            request.messages.pop()
            # remove last tool call message
            tool_call_message = request.messages.pop()
            # add extra tokens before <functioncall> back
            idx = tool_call_message.content.find(FUNCTION_CALL_TOKEN)
            if idx > 0 :
                extra_tokens = tool_call_message.content[:idx]
                response.content = extra_tokens + " " + response.content
            return await self.invoke_function_call(request, request_globals, plugin, response, allow_fallback=False)

        return

    async def verify_results(self, tool_calls: list[ToolCall], plugin_responses: list[DiffbotAPIResponse | DQLResponse]):
        for call, resp in zip(tool_calls, plugin_responses):
            if call.function_name == "dql_v1" and (
                    resp.status != 200 or
                    resp.dict().get("content", {}).get('status') != 200 or
                    len(resp.dict().get("content", {}).get('data', [])) == 0):
                return False
        return True


    async def change_request_to_skip_function_calling(self, tool_call_request):
        tool_call_request.tool_choice = None
        tool_call_request.tools = None
        for message in tool_call_request.messages:
            if message.role == Role.system:
                message.content = self.remove_system_prompt(message.content)
