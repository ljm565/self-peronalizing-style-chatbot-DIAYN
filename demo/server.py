import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
import argparse
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI, HTTPException

from src.tools import Chatter


app = FastAPI()


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=str, default='0', required=False)
parser.add_argument('-c', '--config', type=str, required=False)
parser.add_argument('-m', '--resume_path', type=str, required=False)
args = parser.parse_args()


# Initializer model wrapper
chatter = Chatter(args)


class PromptRequest(BaseModel):
    prompt: str



@app.post("/gpt2")
async def generate_response(request: PromptRequest):
    prompt = request.prompt.strip()
    prompt, response = chatter.generate(prompt)
        
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")
    
    return {"prompt": prompt, "response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8502)