from dataclasses import dataclass
from urllib.parse import urlencode
import json
import httpx
from llm.api_models import ChatCompletionTool, ChatCompletionToolFunctionObject
from models.api import ResponseModel
from services.execute_js import get_js_execution_service

TIMEOUT_SECONDS = 60

class PluginResponse(ResponseModel):
    plugin_url: str
    method: str
    content: object = None

@dataclass
class PluginApiOperationParameter:
    name: str = None
    _in: str = None
    description: str = None
    required: bool = None
    schema: object = None

    @classmethod
    def from_dict(cls, param, api_spec):
        if "schema" in param:
            schema = param.get("schema")
            if "$ref" in schema:
                schema = cls.resolve_ref(schema.get("$ref", api_spec))
                if "title" in schema:
                    del schema["title"]
            if "items" in schema and "$ref" in schema.get("items"):
                schema["items"] = cls.resolve_ref(
                    schema.get("items").get("$ref"), api_spec
                )
                if "title" in schema["items"]:
                    del schema["items"]["title"]
                if "description" in schema["items"]:
                    del schema["items"]["description"]

            return cls(
                name=param.get("name"),
                _in=param.get("in"),
                description=param.get("description"),
                required=param.get("required"),
                schema=schema,
            )

        return None

    @classmethod
    def resolve_ref(cls, ref, api_spec):
        if not ref.startswith("#"):
            raise Exception(f"Unsupported reference: {ref}")

        ref_path = ref[1:].split("/")
        current = api_spec
        for ref_path_part in ref_path:
            if ref_path_part == "":
                continue
            if current is None:
                raise Exception(f"Cannot resolve reference: {ref}")
            current = current.get(ref_path_part)

        return current


@dataclass
class PluginApiOperationRequestBody:
    description: str = None
    required: bool = None
    content: dict[str, object] = None


@dataclass
class PluginApiOperationResponse:
    description: str
    content: dict[str, object]


@dataclass
class PluginApiOperation:
    operation_id: str = None
    server_url: str = None
    api_path: str = None
    method: str = None  # get/post
    description: str = None
    parameters: list[PluginApiOperationParameter] = None
    request_body: PluginApiOperationRequestBody = None
    responses: dict[str, PluginApiOperationResponse] = None


def _get_plugin_spec(plugin_url: str):
    return _get_content(plugin_url)

def _get_content(url: str):
    # get the content from the input URL
    try:
        with httpx.Client(timeout=2) as client:
            response = client.get(url=url)
            return response.json()
    except Exception as e:
        print(f"Error getting content from {url}: {e}")
        return None

class Plugin:
    def __init__(self, plugin_api_spec: dict[str, any]= None):
        self.plugin_apis = self._get_plugin_apis(plugin_api_spec)

    def _get_plugin_apis(self, plugin_api_spec: dict[str, any]):
        # parse the plugin API spec
        if "servers" in plugin_api_spec and len(plugin_api_spec.get("servers")) > 0:
            server_url = plugin_api_spec.get("servers")[0].get("url")
        else:
            server_url = plugin_api_spec["servers"][0]["url"]

        plugin_apis = []
        for api_path in plugin_api_spec.get("paths"):
            api_detail = plugin_api_spec.get("paths")[api_path]
            for method in api_detail:
                operation_object = api_detail[method]
                operation_id = operation_object.get("operationId")
                description = operation_object.get("description")
                parameters = []
                if "parameters" in operation_object:
                    parameters = [
                        PluginApiOperationParameter.from_dict(
                            parameter, plugin_api_spec
                        )
                        for parameter in operation_object.get("parameters")
                    ]

                plugin_apis.append(
                    PluginApiOperation(
                        operation_id=operation_id,
                        server_url=server_url,
                        api_path=api_path,
                        method=method,
                        description=description,
                        parameters=parameters,
                    )
                )

        return plugin_apis

    def _get_tool_from_plugin_api_operation(
        self, pluginApiOperation: PluginApiOperation
    ) -> ChatCompletionTool:
        properties = {}
        required = []
        if pluginApiOperation.parameters:
            for param in pluginApiOperation.parameters:
                properties[param.name] = param.schema
                if param.required:
                    required.append(param.name)

        if pluginApiOperation.request_body:
            for key in pluginApiOperation.request_body:
                for property_key in pluginApiOperation.request_body[
                    key
                ].schema.properties:
                    schema_properties = pluginApiOperation.request_body[
                        key
                    ].schema.properties
                    properties[property_key] = {
                        "type": schema_properties[property_key].type,
                        "description": schema_properties[property_key].description,
                    }
                    if schema_properties[property_key].required:
                        required.append(property_key)

        return ChatCompletionTool(
            type="function",
            function=ChatCompletionToolFunctionObject(
                description=pluginApiOperation.description,
                name=pluginApiOperation.operation_id,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
        )

    def get_tools(self):
        return [self._get_tool_from_plugin_api_operation(operation) for operation in self.plugin_apis]

    async def invoke(self, function_name: str, function_arguments: object, token: str) -> PluginResponse:
        if not function_name or not function_arguments:
            return None

        # call local tools
        if function_name == "execute_js_v1":
            resp = await get_js_execution_service().execute_js(function_arguments["expressions"])
            return PluginResponse(
                plugin_url=function_name, method="INTERNAL", content=resp.json()
            )

        # or call external Diffbot tools
        plugin_apis = [plugin_api for plugin_api in self.plugin_apis if plugin_api.operation_id == function_name]
        if not plugin_apis or len(plugin_apis) == 0:
            return None

        params = {}
        operation = plugin_apis[0]
        if operation.parameters:
            for parameter in operation.parameters:
                if parameter._in == "path" and parameter.name in function_arguments:
                    new_path = function_arguments[parameter.name]
                    operation.api_path = operation.api_path.replace(
                        parameter.name, new_path
                    )
                if parameter._in == "query" and parameter.name in function_arguments:
                    new_query = function_arguments[parameter.name]
                    params[parameter.name] = new_query

        body = {}
        if operation.request_body:
            for key in operation.request_body.content:
                for property_key in operation.request_body.content[key]["schema"][
                    "properties"
                ]:
                    if property_key in function_arguments:
                        body[property_key] = function_arguments[property_key]
            body = json.dumps(body)
        else:
            body = None

        url = f"{operation.server_url}{operation.api_path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        
        print(f"calling plugin api: {url}")

        return await self._call_plugin_api(api_url=url, method=operation.method, body=body, token=token)

    @classmethod
    async def _call_plugin_api(cls, api_url: str, method: str, body: str, token: str):
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            try:
                response = await client.request(
                    method=method,
                    url=api_url,
                    data=body,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return PluginResponse(
                    plugin_url=api_url, method=method, content=response.json()
                )
            except httpx.HTTPStatusError as e:
                status = (
                    "code" in e.response.json()
                    and e.response.json()["code"]
                    or e.response.status_code
                )
                message = (
                    "message" in e.response.json()
                    and e.response.json()["message"]
                    or e.response.reason_phrase
                )
                return PluginResponse(
                    plugin_url=api_url, method=method, status=status, message=message
                )
            except Exception as e:
                return PluginResponse(
                    plugin_url=api_url, method=method, status=500, message=str(e)
                )

plugin = Plugin(_get_plugin_spec("https://llm.diffbot.com/api/.well-known/openapi.yaml"))
def get_plugin() -> Plugin:
    return plugin