# How to Add a Tool to Diffbot LLM Inference

We discuss how to add a tool to Diffbot LLM Inference by showing how to
add the tool `execute_js_v1` for javascript code execution.

### 0. Set up local development

To set up the virtual environment:

```
poetry env use python3.10
poetry shell
poetry install
```

To start vLLM:
 
Self-host one the Diffbot LLM models with docker (see [Self-Hosting](README.md)) and add "-p 8000:8000" to expose 
the vLLM endpoint. Set the vLLM endpoint in config.py.

To start the server: `./start_server.sh`

### 1. Add the new tool to the system prompt (`system_prompt.txt`).

Below is the original system prompt, which includes the definition of the
available tools in javascript.
```
You are a helpful assistant with access to the following functions. Use them if required -
namespace Diffbot {
// Extract the content from the given URLs. Only call this endpoint if the user mentioned a URL.
type extract_v1 = (_: {
// URLs to extract, up to 5
page_url: string[],
}) => any;
// Query the Diffbot Knowledge Graph for an entity or set of entities that match a set of criteria using the Diffbot Query Language syntax.
type dql_v1 = (_: {
// Diffbot Query Language query
dql_query: string,
}) => any;
// Search the web for information that could help answer the user's question.
type web_search_v1 = (_: {
// List of Google advanced search strings (can include phrases, booleans, site:, before:, after:, filetype:, etc)
text: string[],
// Number of results to return (default 5)
num?: number,
// Page number of results to return (default 1)
page?: number,
}) => any;
} // namespace Diffbot
```

To add the tool `execute_js_v1`, we can add the following lines as the last tool:

```
// Execute JavaScript expressions and get accurate results that could help answer the user's question.
type execute_js_v1 = (_: {
// JavaScript expressions to execute separated by newlines
expressions: string,
}) => any;
```

The final result is:

```
You are a helpful assistant with access to the following functions. Use them if required -
namespace Diffbot {
// Extract the content from the given URLs. Only call this endpoint if the user mentioned a URL.
type extract_v1 = (_: {
// URLs to extract, up to 5
page_url: string[],
}) => any;
// Query the Diffbot Knowledge Graph for an entity or set of entities that match a set of criteria using the Diffbot Query Language syntax.
type dql_v1 = (_: {
// Diffbot Query Language query
dql_query: string,
}) => any;
// Search the web for information that could help answer the user's question.
type web_search_v1 = (_: {
// List of Google advanced search strings (can include phrases, booleans, site:, before:, after:, filetype:, etc)
text: string[],
// Number of results to return (default 5)
num?: number,
// Page number of results to return (default 1)
page?: number,
}) => any;
// Execute JavaScript expressions and get accurate results that could help answer the user's question.
type execute_js_v1 = (_: {
// JavaScript expressions to execute separated by newlines
expressions: string,
}) => any;
} // namespace Diffbot
```

### 2. Implement the new tool

See `services/execute_js.py` for the implementation of `execute_js_v1`.

### 3. Call the tool in llm/plugin.py

The `invoke` method is responsible for calling tools requested by the LLM. To call the new tool, we can add the
following lines to this method:

```python
if function_name == "execute_js_v1":
    resp = await get_js_execution_service().execute_js(function_arguments["expressions"])
    return PluginResponse(
        plugin_url=function_name, method="INTERNAL", content=resp.json()
    )
```

where `get_js_execution_service().execute_js()` calls the implementation for this new tool.