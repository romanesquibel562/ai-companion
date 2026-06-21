"""Entrypoint: starts the web dashboard, which runs the vision pipeline in the background.

Dev (no hardware): uvicorn main:app --reload
On the Pi: switch backends to real ones in config.yaml first, then run the same command.
"""
import uvicorn

from companion.config import load_config
from companion.web.server import create_app

config = load_config()
app = create_app(config)

if __name__ == "__main__":
    uvicorn.run(app, host=config.web.host, port=config.web.port)
