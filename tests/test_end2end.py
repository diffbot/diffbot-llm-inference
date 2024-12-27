import contextlib
import time
import threading
import uvicorn
import pytest
from openai import OpenAI
import os


DIFFBOT_TOKEN = os.environ.get("DIFFBOT_TOKEN", "")

class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

@pytest.fixture(scope="session")
def server():
    config = uvicorn.Config("server.main:app", host="127.0.0.1", port=3334, log_level="info")
    server = Server(config=config)
    with server.run_in_thread():
        yield

@pytest.fixture(scope="session")
def endpoint():
    return "http://localhost:3334"

def test_completion_with_tool_call_small(server, endpoint):
    client = OpenAI(api_key=DIFFBOT_TOKEN, base_url=endpoint + "/rag/v1")
    completion = client.chat.completions.create(
        model="diffbot-small",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Who is Nike's new CEO?"
            }
        ]
    )
    print (completion)
    assert 'hill' in completion.choices[0].message.content.lower(), completion.choices[0].message.content

def test_completion_with_tool_call_small_stream(server, endpoint):
    client = OpenAI(api_key=DIFFBOT_TOKEN, base_url=endpoint + "/rag/v1")
    completion = client.chat.completions.create(
        model="diffbot-small",
        temperature=0,
        stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Who is Nike's new CEO?"
            }
        ]
    )
    response = ""
    for chunk in completion:
        if chunk.choices and len(chunk.choices) == 1 and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    print (response)
    assert 'hill' in response.lower(), response

def test_javascript(server, endpoint):
    client = OpenAI(api_key=DIFFBOT_TOKEN, base_url=endpoint + "/rag/v1")
    completion = client.chat.completions.create(
        model="diffbot-small",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": 'What is 3245134 * 3476?'
            }, {
                "role": "tool",
                "content": '<functioncall> {"name": "execute_js_v1", "arguments": {"expressions": "result = 3245134 * 3476; console.log(result)"}}'
            }
        ]
    )
    print(completion)
    answer = completion.choices[0].message.content.lower()
    answer = answer.replace(",","")
    assert '11280085784' in answer, completion.choices[0].message.content
