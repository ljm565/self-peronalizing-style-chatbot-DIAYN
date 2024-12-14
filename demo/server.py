import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
import argparse
from pydantic import BaseModel

import uvicorn
from sconf import Config
from fastapi import FastAPI, HTTPException

from src.tools import Chatter
from src.utils.training_utils import choose_proper_resume_model


app = FastAPI()


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', nargs='+', required=True)
parser.add_argument('-c', '--config', type=str, required=False)
parser.add_argument('-m', '--resume_dir', nargs='+', default=None, type=str, required=False)
parser.add_argument('-l', '--load_model_type', type=str, default='metric', required=False, choices=['metric', 'loss', 'last'])
parser.add_argument('-k', '--model_keys', nargs='+')
args = parser.parse_args()


# Sanity check
if args.resume_dir != None:
    len(args.device) == len(args.resume_dir) == len(args.model_keys)

# Initializer model wrapper
if args.resume_dir != None:
    configs = [Config(os.path.join(path, 'args.yaml')) for path in args.resume_dir]
    resume_paths = [choose_proper_resume_model(directory, args.load_model_type) for directory in args.resume_dir]
    chatters = [Chatter(config, device, resume_path) for config, device, resume_path in zip(configs, args.device, resume_paths)]
else:
    chatters = [Chatter(Config(args.config), args.device[0])]

print(args.model_keys)
class PromptRequest(BaseModel):
    prompt: str



@app.post("/gpt2")
async def generate_response(request: PromptRequest):
    prompt = request.prompt.strip()
    if args.resume_dir == None:
        prompt, response = chatters[0].generate(prompt)
    else:
        results = [chatter.generate(prompt, style_id=1) for chatter in chatters]
        prompt, response = [], []
        for result in results:
            prompt.append(result[0])
            response.append(result[1])
        
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")
    
    return {"prompt": prompt, "response": response, 'key': args.model_keys}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8502)