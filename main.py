import os
import re
import sys
import json
import traceback
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI()

# Enable CORS so the assignment grading script can access your endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

def execute_python_code(code: str) -> dict:
    """
    Executes the user-submitted Python code safely, capturing stdout and tracebacks.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO_capture = sys.modules['io'].StringIO()
    try:
        # Run with an isolated global dictionary context
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        # Capture standard runtime errors
        output = traceback.format_exc()
        return {"success": False, "output": output}
    except SyntaxError:
        # Explicitly capture compilation/syntax errors which lack a full execution traceback
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

def extract_line_numbers_from_traceback(tb_str: str) -> List[int]:
    """
    A robust parser that scans the traceback string for both typical execution
    errors and unique Python SyntaxError line indicators.
    """
    lines = set()
    
    # Pattern 1: Standard runtime tracebacks (e.g., File "<string>", line 3)
    standard_matches = re.findall(r'File\s+"[^"]+",\s+line\s+(\d+)', tb_str)
    for m in standard_matches:
        lines.add(int(m))
        
    # Pattern 2: SyntaxError explicit line summaries (e.g., File "<string>", line 2)
    syntax_matches = re.findall(r'line\s+(\d+)\s*\n', tb_str)
    for m in syntax_matches:
        lines.add(int(m))
        
    return sorted(list(lines))

def analyze_error_with_ai(code: str, error_traceback: str, extracted_lines: List[int]) -> List[int]:
    """
    Uses the AI Pipe OpenRouter proxy to analyze the error context, using
    the extracted line numbers as a precise steering hint.
    """
    aipipe_token = os.environ.get("AIPIPE_TOKEN")
    if not aipipe_token:
        return extracted_lines

    prompt = f"""
    You are a precise Python code debugging tool. Your task is to identify the line number(s) in the original CODE where the error or crash occurred.
    
    CRITICAL ANALYSIS CONTEXT:
    - Traceback string analysis points to line number(s): {extracted_lines}
    - Carefully cross-reference the original code structure below to confirm the precise line number(s) that directly caused or contains the syntax/runtime issue.

    ORIGINAL CODE:
    {code}

    ERROR TRACEBACK:
    {error_traceback}

    Return the final confirmed line number(s) inside the required structured format JSON object.
    """

    url = "https://aipipe.org"
    headers = {
        "Authorization": f"Bearer {aipipe_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "error_lines": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["error_lines"]
            }
        }
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            content_str = data["choices"]["message"]["content"]
            result = json.loads(content_str)
            return result.get("error_lines", extracted_lines)
    except Exception:
        # Fallback to safely parsed line numbers if the network proxy experiences a blip
        return extracted_lines

@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)
    
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }
    else:
        # Deterministically parse out the numbers first
        parsed_lines = extract_line_numbers_from_traceback(execution["output"])
        
        # Invoke AI to validate, reconcile, and format the lines properly
        final_lines = analyze_error_with_ai(request.code, execution["output"], parsed_lines)
        
        return {
            "error": final_lines,
            "result": execution["output"]
        }
