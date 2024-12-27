from typing import List
import tiktoken

from llm.api_models import ChatCompletionRequestMessage, CompletionUsage


def count_stream(stream_response, usage: CompletionUsage):
    for ret in stream_response:
        usage.completion_tokens += 1
        yield ret
    
def count_prompt_tokens(messages: List[ChatCompletionRequestMessage]):
    prompts = "".join([msg.content for msg in messages])
    return num_of_prompt_tokens(prompts)

def num_of_prompt_tokens(prompts: str, encoding_name: str = 'cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(prompts))