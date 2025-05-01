from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Serve UI directory at root
ui_dir = Path(__file__).parent / 'ui'
app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
