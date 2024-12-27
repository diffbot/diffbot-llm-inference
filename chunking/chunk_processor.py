import json, re
import time
from typing import List, Optional, Dict, Any
from collections import deque
from llm.api_models import ChatCompletionRequestMessage
from llm.llms import Role
from chunking.chunking_helper import Chunk, get_similarity_calculator
from models.utils import truncate_long_strings, truncate_long_arrays, truncate_data_dfs
from server.log import get_logstash_logger

logger = get_logstash_logger("chunk_processor")


class DataChunker:
    IDENTIFIER_KEYS = ['name', 'title', 'pageUrl', 'date', 'dql_query']

    @staticmethod
    def tokenize(json_obj: Any) -> int:
        # estimate size instead of expensive json serializing
        if isinstance(json_obj, dict):
            return sum((DataChunker.tokenize(k) + DataChunker.tokenize(v)) for k, v in json_obj.items())
        elif isinstance(json_obj, list):
            return sum(DataChunker.tokenize(item) for item in json_obj)
        elif isinstance(json_obj, str):
            return len(json_obj)
        else:
            return 1

    @staticmethod
    def set_nested_dict(d: Dict[str, Any], path: List[str], value: Any) -> None:
        if not path:
            return
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    @staticmethod
    def get_min_chunk_size(max_chunk_size: int):
        return max(max_chunk_size - 1000, 50)

    @staticmethod
    def filter_urls(markdown_text, max_length=150):
        url_pattern = re.compile(r'\[\[?([^\]]+)\]?\]\(([^)]+)\)')

        def replace_long_urls(match):
            text = match.group(1)
            url = match.group(2)
            if len(url) > max_length:
                return text
            elif '#cite_note' in url:
                return ''
            else:
                return match.group(0)

        return url_pattern.sub(replace_long_urls, markdown_text)

    @staticmethod
    def chunk_text(text: str, max_string_chunk_size: int) -> List[str]:
        text = DataChunker.filter_urls(text)
        min_string_chunk_size = DataChunker.get_min_chunk_size(max_string_chunk_size)
        separators = ["\n\n##", "\n\n", "\n"]
        text_chunks = []
        start_idx = 0
        text_length = len(text)

        while start_idx < text_length:
            end_idx = min(start_idx + max_string_chunk_size + min_string_chunk_size, text_length)
            best_split_idx = None
            start_find_index = start_idx + min_string_chunk_size
            if start_find_index < end_idx:

                for sep in separators:
                    sep_idx = text.rfind(sep, start_find_index, end_idx)
                    if sep_idx != -1:
                        best_split_idx = sep_idx
                        break

            if best_split_idx is None:
                best_split_idx = end_idx

            text_chunks.append(text[start_idx:best_split_idx].strip())
            start_idx = best_split_idx + (
                len(sep) if best_split_idx < text_length and sep in text[best_split_idx:] else 0)

        return text_chunks

    @staticmethod
    def wrap_chunks(chunks: List[Any]) -> List[Chunk]:
        start = time.time()
        chunk_objects = []
        for chunk in chunks:
            if isinstance(chunk, dict) or isinstance(chunk, list):
                chunk_str = json.dumps(chunk, sort_keys=True)
            else:
                chunk_str = chunk
            chunk_objects.append(Chunk(text=chunk_str, size=DataChunker.tokenize(chunk)))
        # print("Time to wrap chunks: ", (time.time() - start) * 1000)
        return chunk_objects

    @staticmethod
    def set_identifiers(chunk: dict, identifiers: dict):
        if not identifiers:
            return
        # each chunk has the identifier values e.g. "name": ".." in case the data is divided across multiple chunks
        for id_key, id_value in identifiers.items():
            # id_value[1] contains the current path used to get the nested path of the value
            DataChunker.set_nested_dict(chunk, id_value[1] + [id_key], id_value[0])

    @staticmethod
    def convert_list_to_dict(data: Any) -> Any:
        # simplify list processing by converting to dict
        if isinstance(data, dict):
            return {k: DataChunker.convert_list_to_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return {
                str(i): DataChunker.convert_list_to_dict(item)
                for i, item in enumerate(data)
            }
        else:
            return data

    @staticmethod
    def convert_dict_to_list(data: Any) -> Any:
        if isinstance(data, dict):
            try:
                sorted_keys = sorted(int(k) for k in data.keys())
                return [
                    DataChunker.convert_dict_to_list(data[str(i)])
                    if str(i) in data else None for i in sorted_keys
                ]
            except ValueError:
                return {k: DataChunker.convert_dict_to_list(v) for k, v in data.items()}
        else:
            return data

    @staticmethod
    def chunk_object(data: Dict[str, Any],
                     max_chunk_size: int, min_chunk_size: int,
                     path: Optional[List[str]] = None,
                     chunks: Optional[List[Any]] = None,
                     identifiers: Optional[Dict] = None):
        path = path or []
        chunks = chunks or [{}]

        if isinstance(data, dict):
            if not identifiers:
                identifiers = {k: (v, path) for k, v in data.items() if k in DataChunker.IDENTIFIER_KEYS}

            for key, value in data.items():
                current_path = path + [key]
                chunk_size = DataChunker.tokenize(chunks[-1])
                size = DataChunker.tokenize({key: value})
                remaining = max_chunk_size - chunk_size

                if size < remaining:
                    DataChunker.set_nested_dict(chunks[-1], current_path, value)
                    DataChunker.set_identifiers(chunks[-1], identifiers)
                else:
                    if chunk_size >= min_chunk_size:
                        new_chunk = {}
                        DataChunker.set_identifiers(new_chunk, identifiers)
                        chunks.append(new_chunk)
                    DataChunker.chunk_object(value, max_chunk_size, min_chunk_size, current_path, chunks, identifiers)

        elif isinstance(data, str):
            data = DataChunker.filter_urls(data)

            # the chunk size reduced by approximated path lengths
            max_string_chunk_size = max_chunk_size - len(path) * 6
            start = time.time()
            string_chunks = DataChunker.chunk_text(data, max_string_chunk_size=max_string_chunk_size)
            # print("chunk_text: ", (time.time()-start)*1000)
            if chunks[-1]:
                chunks.append({})
            for chunk in string_chunks:
                DataChunker.set_identifiers(chunks[-1], identifiers)
                if path:
                    DataChunker.set_nested_dict(chunks[-1], path, chunk)
                else:
                    chunks[-1] = chunk
                chunks.append({})
            if not chunks[-1]:
                chunks.pop()
        else:
            DataChunker.set_nested_dict(chunks[-1], path, data)
        return chunks

    @staticmethod
    def chunk_data(message_input: Any, max_chunk_size: int = 5000) -> List[Any]:
        # Limits to keep truncation time from being unreasonable. We should increase these limits as we improve
        # the truncation performance.
        max_string_length = 500_000
        max_array_length = 100
        max_json_length = 1_000_000
        max_chunks = 500
        try:
            start = time.time()
            if isinstance(message_input, str) and (message_input.startswith("{") or message_input.startswith("[")):
                # TODO: loading the whole json takes ~70ms for JSON string with 40M+ chars.
                #  Try json streaming to stop reading after a particular limit?
                message_input = json.loads(message_input)
            # print("time to load json: ", (time.time() - start)*1000)
            start = time.time()
            message_input = truncate_long_strings(message_input, max_string_length=max_string_length)
            message_input = truncate_long_arrays(message_input, max_array_length=max_array_length)
            message_input, _ = truncate_data_dfs(message_input, max_length=max_json_length)
            # print("dumb truncation: ", (time.time() - start) * 1000)
            start = time.time()
            message_input = DataChunker.convert_list_to_dict(message_input)
            # print("convert list to dict: ", (time.time() - start) * 1000)
            start = time.time()
            processed_chunks = DataChunker.chunk_object(message_input, max_chunk_size=max_chunk_size,
                                                        min_chunk_size=DataChunker.get_min_chunk_size(max_chunk_size))
            processed_chunks = processed_chunks[:max_chunks]
            # print("chunk_object: ", (time.time()-start)*1000)
        except (TypeError, NameError, ValueError, OverflowError) as e:
            start = time.time()
            logger.error(f"Exception during chunk_data: {e}", exc_info=True)
            if not message_input:
                message_input = ""
            if not isinstance(message_input, str):
                message_input = str(message_input)
            message_input = message_input[:max_string_length]
            processed_chunks = DataChunker.chunk_text(message_input, max_string_chunk_size=max_chunk_size)
            # print("chunk exception: ", (time.time() - start) * 1000)
        return DataChunker.wrap_chunks(processed_chunks)


class ChunkProcessor:
    def __init__(self):
        self.data_chunker = DataChunker()

    @staticmethod
    def get_last_user_message(messages):
        last_user_message_index = None
        last_user_message = None
        for i in range(len(messages) - 1, -1, -1):
            # make sure that the user message is not a tool response
            if messages[i].role == Role.user and (
                    i == 0 or ('<functioncall>' not in messages[i - 1].content or messages[i - 1].role == 'system')) and \
                    not messages[i].content.startswith('{"status": '):
                last_user_message_index = i
                last_user_message = messages[i]
                break
        return last_user_message_index, last_user_message

    def merge_chunks(self, chunks):
        def merge_json(json1, json2):
            merged = {}
            seen_key = set()
            # keep order of keys
            key_list = []
            key_list.extend([k for k, v in json1.items()])
            key_list.extend([k for k, v in json2.items()])
            for key in key_list:
                if key in seen_key:
                    continue
                seen_key.add(key)
                if key in json1 and key in json2:
                    if isinstance(json1[key], dict) and isinstance(json2[key], dict):
                        if json1[key] == json2[key]:
                            merged[key] = json1[key]
                        else:
                            merged[key] = merge_json(json1[key], json2[key])
                    elif isinstance(json1[key], str) and isinstance(json2[key], str):
                        if json1[key] == json2[key]:
                            merged[key] = json1[key]
                        else:
                            merged[key] = f"{json1[key]} {json2[key]}"
                    else:
                        merged[key] = json2[key]
                elif key in json1:
                    merged[key] = json1[key]
                else:
                    merged[key] = json2[key]
            return merged

        def try_parse_json(s):
            try:
                if not s.endswith('}'):
                    return s, False
                return json.loads(s), True
            except json.JSONDecodeError:
                return s, False

        merged_json = {}
        merged_str = ""

        for s in chunks:
            parsed, is_json = try_parse_json(s)
            if is_json:
                if merged_json:
                    merged_json = merge_json(merged_json, parsed)
                else:
                    merged_json = parsed
            else:
                if merged_str:
                    merged_str += "\n" + s
                else:
                    merged_str = s

        if merged_json:
            if merged_str:
                return json.dumps(merged_json) + "\n" + merged_str
            else:
                return json.dumps(DataChunker.convert_dict_to_list(merged_json))
        else:
            return merged_str

    # this method should take the message array as input
    # iterate messages in the order - last user message
    # Chunk each message content
    # Compute similarity of all chunks with the last user role content and pick the most similar chunks
    def process_messages(self, messages: List[ChatCompletionRequestMessage], max_tokens: int,
                         max_tokens_last_tool: int, model_input_token_limit: int = 12000, log_ctx: dict = {}) -> List[
        ChatCompletionRequestMessage]:
        max_size = max_tokens * 4
        max_size_last_tool = max_tokens_last_tool * 4
        model_input_token_limit_size = model_input_token_limit * 4

        if max_size_last_tool > max_size:
            raise Exception("max_size_last_tool cannot be greater than max_size")
        max_chunk_size = 2500
        log_ctx["max_size"] = max_size
        log_ctx["max_chunk_size"] = max_chunk_size
        try:

            # returns immediately in case under limit: identity function
            total_content_size = sum(len(message.content) for message in messages)
            if total_content_size < max_size:
                return messages

            _, last_user_message = self.get_last_user_message(messages)

            # TODO: this can potentially return more tokens than max_tokens. Write an alternative algorithm
            #  (e.g., latest messages)?
            if not last_user_message:
                log_ctx["error"] = "could not find last user message"
                return messages

            # no processing possible if the user message is too long; return messages,
            # UI should return error
            if len(last_user_message.content) >= max_size:
                # TODO: truncate last user message by keeping beginning and end, while ignoring the middle?
                log_ctx["error"] = "user message is too long"
                return messages

            start_chunking = time.time()
            chunks_role_list = []
            last_user_message_index = None
            system_message = None
            for message in messages:
                if message.role == Role.system:
                    system_message = message
                elif message == last_user_message:
                    # include the last user message in the
                    chunk = Chunk(text=message.content, size=len(message.content), include_in_request=True,
                                  similarity=0)
                    chunks_role_list.append(([chunk], message.role, False))
                else:
                    is_intext_functioncall = False
                    if message.role == Role.assistant and message.content.startswith('<functioncall>'):
                        is_intext_functioncall = False
                    elif message.role == Role.assistant and not message.content.startswith(
                            '<functioncall>') and '<functioncall>' in message.content:
                        # flag to identify if there is only functioncall in the assistant message or functioncall exists in text with response
                        is_intext_functioncall = True
                    chunks_role_list.append(
                        (self.chunk_data(message.content, max_chunk_size), message.role, is_intext_functioncall))

            all_chunks: List[Chunk] = []
            for chunk_list, role, is_intext_functioncall in chunks_role_list:
                for chunk in chunk_list:
                    chunk.role = role
                    chunk.is_intext_functioncall = is_intext_functioncall
                    all_chunks.append(chunk)
                    if last_user_message.content == chunk.text:
                        last_user_message_index = len(all_chunks) - 1

            if last_user_message_index is None:
                last_user_message_index = len(all_chunks)
            log_ctx["chunking_time"] = round((time.time() - start_chunking) * 1000)
            log_ctx["num_chunks"] = len(all_chunks)
            # print("chunking_time: ", log_ctx["chunking_time"])
            # print("num_chunks: ", log_ctx["num_chunks"])

            start_similarity = time.time()
            similarity_calculator = get_similarity_calculator()
            similarity_calculator.calculate_similarity(last_user_message.content, all_chunks, explain=False)
            log_ctx["similarity_time"] = round((time.time() - start_similarity) * 1000)
            # print("similarity_time: ", log_ctx["similarity_time"])

            last_assistant_message_index = -1
            last_tool_messsage_index = -1
            last_function_call_index = -1
            prev_role = None

            # Hardcode the similarity to 1 the beginning of any assistant message after that (not functioncalls) to prioritize these
            for i in range(last_user_message_index, len(all_chunks)):
                # Keep the assistant message which does not start with functioncall or has intext functioncall
                if all_chunks[i].role == Role.assistant and (
                        not all_chunks[i].text.startswith('<functioncall>') or all_chunks[i].is_intext_functioncall):
                    all_chunks[i].similarity = 1
                    if prev_role != all_chunks[i].role:  # first chunk of this assistant language turn
                        last_assistant_message_index = i
                if all_chunks[i].role == Role.assistant and prev_role != all_chunks[i].role:
                    last_function_call_index = i
                if all_chunks[i].role != Role.assistant and prev_role != all_chunks[i].role:
                    last_tool_messsage_index = i
                prev_role = all_chunks[i].role
            # add tool call + response pairs in case of interleaved calls into history messages, only keep the last pair
            # move old interleaving messages to history
            old_interleaving_chunks = []
            current_chunks = all_chunks[last_user_message_index:]
            if last_assistant_message_index != -1:
                old_interleaving_chunks = all_chunks[last_user_message_index + 1:last_assistant_message_index]
                current_chunks = [all_chunks[last_user_message_index]] + all_chunks[last_assistant_message_index:]

            final_messages: deque[ChatCompletionRequestMessage] = deque()
            total_length_so_far = 0
            if system_message:
                total_length_so_far += len(system_message.content)

            expand_context_size, max_size_last_tool = self.expand_last_tool_context(all_chunks,
                                                                                    last_function_call_index,
                                                                                    last_tool_messsage_index,
                                                                                    max_size_last_tool,
                                                                                    model_input_token_limit_size)

            # choose current chunks, that is, those containing the last user message and last tool response
            # use at least half of the remaining room with current chunks
            min_length = total_length_so_far + round((max_size_last_tool - total_length_so_far) / 2)

            total_length_so_far = self.convert_best_chunks_to_messages(final_messages, current_chunks,
                                                                       total_length_so_far, min_length,
                                                                       max_size_last_tool,
                                                                       similarity_calculator.threshold)
            if expand_context_size > 0:
                # adjust the max size to make up for the expansion of the tool messages size
                max_size += min(total_length_so_far, expand_context_size)
                max_size = min(max_size, model_input_token_limit_size)

            # choose history messages
            history_chunks = all_chunks[:last_user_message_index] + old_interleaving_chunks
            history_chunks = [c for c in history_chunks if c.role != Role.system]
            # inverse history chunks so that we prioritize those close to the last user message
            history_chunks = history_chunks[::-1]
            history_messages: list[ChatCompletionRequestMessage] = list()
            self.convert_best_chunks_to_messages(history_messages, history_chunks,
                                                 total_length_so_far, min_length=max_size,
                                                 max_length=max_size,
                                                 threshold=similarity_calculator.threshold)

            for index in range(0, len(history_messages), 2):
                remaining = max_size - total_length_so_far
                if index + 1 < len(history_messages):
                    content_length = len(history_messages[index].content) + len(history_messages[index + 1].content)
                else:
                    content_length = len(history_messages[index].content)
                if content_length >= remaining:
                    break
                # adding history messages in pairs to avoid errors
                final_messages.appendleft(ChatCompletionRequestMessage(content=history_messages[index].content,
                                                                       role=history_messages[index].role))
                if index + 1 < len(history_messages):
                    final_messages.appendleft(ChatCompletionRequestMessage(content=history_messages[index + 1].content,
                                                                           role=history_messages[index + 1].role))
                total_length_so_far += content_length
            if system_message:
                final_messages.appendleft(
                    ChatCompletionRequestMessage(content=system_message.content, role=Role.system))
            return list(final_messages)

        except Exception as e:
            logger.error(f"Exception while truncating messages: {e}", exc_info=True)
            return self.backup_process_messages(messages, max_size=max_size, max_chunk_size=max_chunk_size,
                                                max_chunks=20)

    def expand_last_tool_context(self, all_chunks, last_function_call_index, last_tool_messsage_index,
                                 max_size_last_tool, model_input_token_limit_size) -> int:
        expand_needed = False
        if last_tool_messsage_index != -1 and last_function_call_index != -1:
            # merge the last assistant message into a single message
            last_tool_message_content = []
            for i in range(last_function_call_index, last_tool_messsage_index):
                last_tool_message_content.append(all_chunks[i].text)
            last_tool_message_content = self.merge_chunks(last_tool_message_content)
            if last_tool_message_content.count('<functioncall>') == 1:
                functioncall = last_tool_message_content.index('<functioncall>')
                functioncall_content = last_tool_message_content[functioncall + len('<functioncall>'):].strip()
                if functioncall_content.startswith('{') and functioncall_content.endswith('}'):
                    try:
                        # check if the functioncall is for extract_v1
                        functioncall_content = json.loads(functioncall_content)
                        if functioncall_content['name'] == 'extract_v1':
                            expand_needed = True
                    except:
                        pass
        expand_content_size = 0
        if expand_needed:
            # increase the max size of the last tool response to model limit/2
            new_max_size_last_tool = model_input_token_limit_size // 2
            expand_content_size = new_max_size_last_tool - max_size_last_tool
            expand_content_size = 0 if expand_content_size < 0 else expand_content_size
            if new_max_size_last_tool > max_size_last_tool:
                max_size_last_tool = new_max_size_last_tool

        return expand_content_size, max_size_last_tool

    def convert_best_chunks_to_messages(self, messages, chunks, total_length_so_far, min_length, max_length, threshold):
        self.choose_chunks(chunks, max_length, min_length, threshold, total_length_so_far)
        # collect messages
        first_chunk_of_message = None
        combined_text = []
        for i in range(0, len(chunks)):
            content_length = len(chunks[i].text)
            if content_length > (max_length - total_length_so_far):
                break
            if chunks[i].include_in_request:
                combined_text.append(chunks[i].text)
                total_length_so_far += content_length
            # store the first chunk of a message for future use
            if i == 0 or chunks[i].role != chunks[i - 1].role:
                first_chunk_of_message = i

            is_last_chunk = (i + 1 >= len(chunks)  # last chunk overall
                             or chunks[i].role != chunks[i + 1].role  # last chunk for this role
                             or len(chunks[i + 1].text) > (max_length - total_length_so_far)  # next chunk doesn't fit
                             )
            if is_last_chunk:
                # this is the last chunk of a message, merge chosen chunks
                if not combined_text:
                    # if no chunk was selected via similarity for current message, chose the first chunk.
                    combined_text = [chunks[first_chunk_of_message].text]
                    total_length_so_far += len(chunks[first_chunk_of_message].text)
                messages.append(ChatCompletionRequestMessage(content=self.merge_chunks(combined_text),
                                                             role=chunks[i].role))
                combined_text = []

        return total_length_so_far

    def choose_chunks(self, chunks, max_length, min_length, threshold, total_length_so_far):
        estimated_remaining = max_length - total_length_so_far
        # sort chunks by similarity
        sorted_indices = sorted(range(len(chunks)), key=lambda k: chunks[k].similarity, reverse=True)
        min_length = min_length
        for i, index in enumerate(sorted_indices):
            content_length = len(chunks[index].text)
            if content_length > estimated_remaining:
                break
            if chunks[index].similarity > threshold or (max_length - estimated_remaining) < min_length:
                chunks[index].include_in_request = True
                estimated_remaining = estimated_remaining - content_length

    def chunk_data(self, data, max_chunk_size):
        return self.data_chunker.chunk_data(data, max_chunk_size=max_chunk_size)

    def backup_process_messages(self, messages: List[ChatCompletionRequestMessage], max_size: int, max_chunk_size: int,
                                max_chunks: int) -> List[ChatCompletionRequestMessage]:

        processed_messages: deque[ChatCompletionRequestMessage] = deque()

        last_user_message_index, _ = self.get_last_user_message(messages)

        # TODO: this can potentially return more tokens than max_tokens
        if last_user_message_index is None:
            return messages

        current_size = 0
        system_message = None
        if len(messages) > 0 and messages[0].role == Role.system:
            current_size = len(messages[0].content)
            system_message = messages[0]

        # first should be the last user message, all following messages, then preceding history messages till the limit is reached
        for i in range(last_user_message_index, len(messages)):
            remaining = max_size - current_size
            content_length = len(messages[i].content)
            message_length = min(max_chunk_size * 5, content_length)
            if message_length >= remaining or len(processed_messages) >= max_chunks:
                break
            messages[i].content = messages[i].content[:message_length]
            processed_messages.append(messages[i])
            current_size += len(messages[i].content[:message_length])

        for i in range(last_user_message_index - 1, -1, -1):
            remaining = max_size - current_size
            content_length = len(messages[i].content)
            message_length = min(max_chunk_size, content_length)
            if messages[i].role == Role.system:
                continue
            if message_length >= remaining or len(processed_messages) >= max_chunks:
                # every assistant message should have user message pair
                if processed_messages[0].role == Role.assistant:
                    processed_messages.popleft()
                break
            messages[i].content = messages[i].content[:message_length]
            processed_messages.appendleft(messages[i])
            current_size += len(messages[i].content[:message_length])

        if system_message:
            processed_messages.appendleft(system_message)

        return list(processed_messages)


chunking_processor = None


def get_chunking_processor():
    global chunking_processor
    if not chunking_processor:
        chunking_processor = ChunkProcessor()
    return chunking_processor

