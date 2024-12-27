from abc import ABC
from types import GeneratorType
from typing import Any, List, Optional, Dict, Union
from enum import Enum
from dataclasses import dataclass

from llm.api_models import ChatCompletionRequestMessage, CompletionUsage, CreateChatCompletionRequest, \
    ChatCompletionResponse, ChatCompletionStreamResponse, RequestGlobals

# # supported LLM models and functions

OPENAI_GPT_3_5 = 'gpt-3.5-turbo'
OPENAI_GPT_4 = 'gpt-4-turbo'
MISTRAL_7B_INSTRUCT = 'mistral-7b-instruct-32k'
OUTPUT_TOKENS_RESERVE = 4000
INPUT_TOKENS_LIMIT = 16384 # no model should have a tokenLimit lower than INPUT_TOKENS_LIMIT + OUTPUT_TOKENS_RESERVE.
LAST_TOOL_TOKENS_LIMIT = 6000 # limit on tokens to use for last tool response. should be lower than INPUT_TOKENS_LIMIT.

class ModelID(str, Enum):
    DIFFBOT_SMALL = 'diffbot-small'
    UNKNOWN = 'unknown'

    @classmethod
    def get_model_id(cls, model_name: str):
        try:
            return ModelID(model_name)
        except ValueError:
            return ModelID.UNKNOWN

@dataclass
class ModelInfo:
    id: ModelID
    model: str
    tokenLimit: int

LLMS = {
    ModelID.DIFFBOT_SMALL: ModelInfo(
        id=ModelID.DIFFBOT_SMALL,
        model=ModelID.DIFFBOT_SMALL,
        tokenLimit=131072,
    ),
}

# # Chat related data structures.
class Role(str, Enum):
    system = 'system'
    user = 'user'
    assistant = 'assistant'
    tool = 'tool'

class LLM(ABC):
    @classmethod
    def is_supported(cls, model: ModelID) -> bool:
        pass
    
    def generate_prompt(self, system_prompt: str, messages: List[ChatCompletionRequestMessage], maxLength: int) -> str:
        pass
    
    def chat_completion(self, request: CreateChatCompletionRequest, request_globals: RequestGlobals) \
            -> Union[ChatCompletionResponse, ChatCompletionStreamResponse]:
        pass
