from abc import ABC
import asyncio
import json
import os
import tempfile
from typing import Dict, Any

from models.api import JSExecutionResponse


class JSExecutionService(ABC):

    def __init__(self) -> None:
        super().__init__()

    async def execute_js_code(self, js_code: str, timeout: int) -> Dict[str, Any]:
        # Create temporary file for the JS code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            # Disable potentially harmful functions by replacing with empty objects
            safe_environment = """
                var require = {};
                var process = {};
                var global = {};
            """

            # Wrap the code to capture console.log output and the final result
            wrapped_code = f"""
                try {{
                    var logs = [];
                    _consolelog = console.log;
                    console.log = function(...args) {{logs.push(args.map(String).join(" "));}};
                    var result = eval({repr(js_code)});
                    _consolelog(JSON.stringify({{logs: logs, result: result}}));
                }} catch (error) {{
                    _consolelog(JSON.stringify({{error: error.toString()}}));
                }}
            """

            # Write the complete code to the temp file
            f.write(f"""
                {safe_environment}
                {wrapped_code}
            """)
            temp_path = f.name

        try:
            # Create subprocess to run Node.js
            process = await asyncio.create_subprocess_exec(
                'node', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                # Wait for the process with timeout
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                raise Exception("JavaScript execution timed out!")

            # Clean up temp file
            os.unlink(temp_path)

            if stderr:
                stderr_text = stderr.decode().strip()
                if stderr_text:
                    raise Exception(f"JavaScript error: {stderr_text}")

            # Parse the output
            output = stdout.decode().strip()
            if not output:
                raise Exception("No output from JavaScript code")

            try:
                result = json.loads(output)
                if "error" in result:
                    raise Exception(result["error"])
                return {"logs": result.get("logs", []), "result": result.get("result")}
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON output: {output}")

        except Exception as e:
            # Clean up temp file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def clean_code_block(self, code: str) -> str:
        if code.startswith('```javascript') and code.endswith('```'):
            code = code[len('```javascript'): -3].strip()
        elif code.startswith('```') and code.endswith('```'):
            code = code[3:-3].strip()
        if code.startswith('<script>') and code.endswith('</script>'):
            code = code[len('<script>'): -len('</script>')].strip()
        return code

    async def execute_js(self, expressions: str = None, timeout: int = 2) -> JSExecutionResponse:
        try:
            if not expressions:
                raise Exception("No JavaScript code provided")

            result = await self.execute_js_code(self.clean_code_block(expressions), timeout)

            if not result:
                raise Exception("No result returned from JavaScript code")

            return JSExecutionResponse(expression=expressions, status=200, message="Success", data=result)

        except Exception as e:
            return JSExecutionResponse(expression=expressions, status=500, message=str(e))


js_execution_service = None


def get_js_execution_service() -> JSExecutionService:
    global js_execution_service
    if js_execution_service is None:
        js_execution_service = JSExecutionService()
    return js_execution_service
