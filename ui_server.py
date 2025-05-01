from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()
# Serve wave/ui at root
ui_dir = Path(__file__).parent / 'wave' / 'ui'
app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
