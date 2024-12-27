from typing import Optional, Union, AsyncIterator

from llm.diffbot_tool_call_llm import DiffbotToolCallLLM, get_diffbot_tool_call_llm
from llm.api_models import CompletionUsage, CreateChatCompletionRequest, ChatCompletionResponse, \
    ChatCompletionStreamResponse, LLMException, Error, ChatCompletionRequestMessage, RequestGlobals
from llm.llms import ModelID, LLMS, LLM, Role
from llm.tool_call_llm import ToolCallLLM
from llm.plugin import Plugin, get_plugin

class KgRagService:
    """
    Service to rag against Diffbot Knowledge Graph.
    """
    def __init__(self) -> None:
        self.diffbot_models = {ModelID.DIFFBOT_SMALL}

    @classmethod
    def get_llm(cls, model_id: ModelID) -> LLM:
        if DiffbotToolCallLLM.is_supported(model_id):
            return get_diffbot_tool_call_llm()
        else:
            raise LLMException(error=Error(code=429, message=f"unsupported model: {model_id}", param=None, type=None))
    
    @classmethod
    def get_tool_call_llm(cls, model_id: ModelID) -> ToolCallLLM:
        if DiffbotToolCallLLM.is_supported(model_id):
            return get_diffbot_tool_call_llm()
        else:
            raise LLMException(error=Error(code=429, message=f"unsupported tool call model: {model_id}", param=None, type=None))

    async def chat_completions(self,
                               request: CreateChatCompletionRequest,
                               plugin: Optional[Plugin] = get_plugin(),
                               request_globals: RequestGlobals = RequestGlobals(),
                               ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        model_id = ModelID.get_model_id(request.model)
        request.model = LLMS[model_id].model
        llm = self.get_llm(model_id)

        if (model_id not in self.diffbot_models) or (request.tool_choice == "none"):
            # only diffbot models support tool calling
            print(f"calling {request.model} without tool calling.")
            if llm:
                request.tool_choice = None
                return await llm.chat_completion(request=request, request_globals=request_globals)
            else:
                raise LLMException(error=Error(code=429, message=f"llm not initialized: {request.model}", param=None, type=None))
        else:
            # Using a "diffbot-" model

            print(f"calling {request.model} with tool calling")

            if request.tool_choice is None:
                request.tool_choice = "auto"

            request.tools = plugin.get_tools() #TODO: call to llm.diffbot.com and cache?
            tool_call_llm = self.get_tool_call_llm(model_id)
            llm_result = await tool_call_llm.process_tool_calls(request, plugin, request_globals)
            return llm_result

async def combine_stream(non_stream_result, stream_result):
    if non_stream_result:
        yield non_stream_result
    async for item in stream_result:
        yield item

async def skip_diffbot_responses(llm_result):
    async for item in llm_result:
        if isinstance(item, ChatCompletionStreamResponse):
            yield item

kg_rag_service = None
def get_kg_rag_service() -> KgRagService:
    global kg_rag_service
    if kg_rag_service is None:
        kg_rag_service = KgRagService()
        
    return kg_rag_service