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

# 6. AI Agent Function: Asks Gemini to read the code and error traceback
def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    # Fetch your AI Pipe token securely from environment variables
    aipipe_token = os.environ.get("AIPIPE_TOKEN")
    
    # Configure the client to point to the AI Pipe OpenRouter proxy
    client = genai.Client(
        api_key=aipipe_token,
        http_options={"api_version": "v1", "base_url": "https://aipipe.org/openrouter/v1"}
    )

    prompt = f"""
    Analyze this Python code and its error traceback.
    Identify the line number(s) where the error occurred.

    CODE:
    {code}

    TRACEBACK:
    {error_traceback}

    Return the line number(s) where the error is located.
    """

    # Request structured output from the specific OpenRouter-Gemini model
    response = client.models.generate_content(
        model='google/gemini-2.0-flash-lite-001',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER)
                    )
                },
                required=["error_lines"]
            )
        )
    )

    # Parse and safely extract the list of line numbers
    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines

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

