import time
from openai import AsyncOpenAI
from typing import Union, AsyncIterator
from llm.token_utils import count_prompt_tokens

from models.utils import cleanNullValues
from config import Config, get_config
from llm.llms import LLM, ModelID, LLMS, INPUT_TOKENS_LIMIT, OUTPUT_TOKENS_RESERVE, LAST_TOOL_TOKENS_LIMIT
from llm.api_models import ChatCompletionRequestMessage, CompletionUsage, CreateChatCompletionRequest, \
    ChatCompletionResponse, ChatCompletionStreamResponse, LLMException, RequestGlobals, Error
from chunking.chunk_processor import get_chunking_processor
from server.log import get_logstash_logger

logger = get_logstash_logger("openai_gpt")

class OpenAIModel(LLM):
    supported_model = {ModelID.DIFFBOT_SMALL}
    diffbot_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=get_config().get_vllm_server_url()+"/v1",
    )

    @classmethod
    def is_supported(cls, model: ModelID) -> bool:
        return model and model in cls.supported_model
    
    async def chat_completion(self, request: CreateChatCompletionRequest, request_globals: RequestGlobals) \
            -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        model_token_limit = LLMS[request.model].tokenLimit
        if model_token_limit < INPUT_TOKENS_LIMIT + OUTPUT_TOKENS_RESERVE:
            input_tokens_limit = model_token_limit - OUTPUT_TOKENS_RESERVE
        else:
            input_tokens_limit = INPUT_TOKENS_LIMIT
        log_ctx = {}
        start_truncation = time.time()
        json_length_before = sum(len(message.content) for message in request.messages)
        request.messages = get_chunking_processor().process_messages(request.messages,
                                                                     max_tokens=input_tokens_limit,
                                                                     max_tokens_last_tool=LAST_TOOL_TOKENS_LIMIT,
                                                                     log_ctx=log_ctx)
        json_length_after = sum(len(message.content) for message in request.messages)
        truncation_time = round((time.time() - start_truncation) * 1000) #ms
        log_ctx.update({"truncation_time": truncation_time,
                        "json_length_before": json_length_before,
                        "json_length_after": json_length_after,
                        "is_longer_after_truncation": json_length_after > json_length_before,
                        "is_longer_than_limit": json_length_after > (log_ctx.get("max_size", INPUT_TOKENS_LIMIT * 4)),
                        "model": request.model
                        })
        logger.info("message truncation completed", extra=log_ctx)

        request_new = cleanNullValues(request.dict())

        if request_new['model'] not in request_globals.usage:
            request_globals.usage[request_new['model']] = CompletionUsage()
        curr_usage = request_globals.usage[request_new['model']]

        if request_new['model'] == ModelID.DIFFBOT_SMALL:
            client = self.diffbot_client
            request_new['model'] = 'fine-tuned'

            # VLLM errs with these unnecessary fields, remove them
            request_new.pop('tools', None)
            request_new.pop('tool_choice', None)
            for message in request_new['messages']:
                # Convert all role:tool to role:user for VLLM to enforce alternative user/assisant turns
                if message['role'] == "tool":
                    message['role'] = "user" 
                message.pop('tool_call_id', None)

            # stop tokens
            request_new['stop'] = ['<|endoftext|>', '<|im_end|>',
                                   # to prevent the LLM from having infinite conversations with itself
                                   '### USER:', '### ASSISTANT:' , '### <functioncall>'
                                   ]
        else:
            raise LLMException(error=Error(code=422, message="Invalid model: {}".format(request_new['model'])))

        request_globals.internal_request = request_new # save last internal request
        response = await client.chat.completions.create(**request_new, timeout=60)

        if request.stream:
            # get prompt_token
            prompt_tokens_count = count_prompt_tokens(request.messages)
            curr_usage.prompt_tokens += prompt_tokens_count
            curr_usage.total_tokens += prompt_tokens_count
            return parse_stream(response, curr_usage)
        else:
            chat_response = ChatCompletionResponse(**response.dict())
            curr_usage.prompt_tokens += chat_response.usage.prompt_tokens
            curr_usage.completion_tokens += chat_response.usage.completion_tokens
            curr_usage.total_tokens += chat_response.usage.total_tokens
            
            return chat_response

async def parse_stream(response, curr_usage: CompletionUsage):
    async for event in response:
        if not event.object:  # skip empty events
            continue
        if event.choices and len(event.choices) == 1 and event.choices[0].delta and event.choices[0].delta.content is None:
            event.choices[0].delta.content = ""
        curr_usage.completion_tokens += 1
        curr_usage.total_tokens += 1
        yield ChatCompletionStreamResponse(**event.dict())

openai_model = None
def get_openai_llm():
    global openai_model
    if not openai_model:
        openai_model = OpenAIModel()
    return openai_model