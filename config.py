import os

SERVER = "http://YOUR_SERVER_HERE:3333"
VLLM_SERVER = "http://localhost:8000"
SYSTEM_PROMPT_FILE="system_prompt.txt"

class Config:

    def __init__(self, server, vllm_server, system_prompt_file) -> None:
        self.server = server
        self.vllm_server = vllm_server
        self.system_prompt_file = system_prompt_file
        with open(self.system_prompt_file, "r") as f:
            self.system_prompt=f.read()

    def get_server_url(self):
        return self.server

    def get_vllm_server_url(self):
        return self.vllm_server

    def get_system_prompt(self):
        return self.system_prompt

config = None
def get_config():
    global config
    if config is not None:
        return config

    server = os.environ.get("SERVER", None)
    if not server:
        server=SERVER
    vllm_server = os.environ.get("VLLM_SERVER", None)
    if not vllm_server:
        vllm_server = VLLM_SERVER
    system_prompt_file = os.environ.get("SYSTEM_PROMPT_FILE", None)
    if not system_prompt_file:
        system_prompt_file = SYSTEM_PROMPT_FILE
    config = Config(server=server, vllm_server=vllm_server, system_prompt_file=system_prompt_file)
    return config