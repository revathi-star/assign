import os
import sys
from io import StringIO
import traceback
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Key Requirement: Enable CORS so grading/testing scripts can reach it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Define the Request format (What the user sends)
class CodeRequest(BaseModel):
    code: str

# 4. Define the Structured Response for the AI
class ErrorAnalysis(BaseModel):
    error_lines: List[int]

# 5. Tool Function: Safely runs the submitted code block
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})  # Using an isolated empty dictionary for safety
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

import httpx
import json
import os
from typing import List

def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    aipipe_token = os.environ.get("AIPIPE_TOKEN")
    
    if not aipipe_token:
        print("ERROR: AIPIPE_TOKEN environment variable is not set!")
        return []

    # OPTIMIZED PROMPT: Forces the LLM to follow deterministic extraction rules
    prompt = f"""
    You are a precise Python debugging tool. Your task is to extract the exact line number where the runtime error or syntax error occurred by analyzing the provided TRACEBACK.

    CRITICAL INSTRUCTIONS:
    1. Read the TRACEBACK from the bottom up.
    2. Look for patterns like 'File "<string>", line X' or 'File "<stdin>", line X' inside the TRACEBACK string. The number X is the exact line number where the error occurred.
    3. Do NOT guess line 1 unless the traceback explicitly points to line 1.
    4. Base your response purely on the line numbers explicitly present in the traceback string.

    CODE TO REFERENCE:
    {code}

    TRACEBACK TO ANALYZE:
    {error_traceback}

    Return the line number(s) in the required structured output schema.
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
            content_str = data["choices"][0]["message"]["content"]
            result = json.loads(content_str)
            
            return result.get("error_lines", [])
    except Exception as e:
        print(f"AI Analysis Failed: {str(e)}")
        return []



# 7. Create the POST Endpoint required by the task
@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    # Run the provided snippet
    execution = execute_python_code(request.code)
    
    # Flow Step 2: Check if code succeeded
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }
    else:
        # Flow Step 3 & 4: Only invoke AI if there is an error
        detected_lines = analyze_error_with_ai(request.code, execution["output"])
        return {
            "error": detected_lines,
            "result": execution["output"]
        }

