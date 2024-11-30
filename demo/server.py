import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))

import uvicorn
import argparse
from fastapi import FastAPI, WebSocket

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





@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await chatter.generate(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)