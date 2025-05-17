import pytest
# skip if dash testing or multiprocess or uvicorn missing
pytest.importorskip("dash.testing.application_runners")
pytest.importorskip("multiprocess")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.integration
import requests
from dash.testing.application_runners import import_app
from pathlib import Path

def test_ui_load_and_save(tmp_path, dash_duo, monkeypatch):
    # setup config and dummy API
    cfg = {'symbols': ['S'], 'timeframes': [{'tf': 'T', 'windows': [1]}],
           'pattern_api': {'host': '127.0.0.1', 'port': 9999}}
    with open(tmp_path / 'config.yml', 'w') as f:
        import yaml
        yaml.dump(cfg, f)
    # dummy FastAPI server
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI()
    @app.get('/catalog')
    def cat(): return JSONResponse({'patterns':[{'pattern_id':'p1','label':'L','color':'#f00'}]})
    @app.put('/patterns/p1')
    def upd(): return JSONResponse({'status':'ok'})
    # run dummy API
    import threading, uvicorn
    server = threading.Thread(target=uvicorn.run, args=(app,), kwargs={'host':'127.0.0.1','port':9999}, daemon=True)
    server.start()
    # launch UI
    monkeypatch.chdir(tmp_path)
    ui = import_app('wave.ui.app').app
    dash_duo.start_server(ui)
    dash_duo.wait_for_element('#pattern-table')
    # verify load
    table = dash_duo.find_element('#pattern-table')
    assert 'p1' in table.text
    # edit cell
    cell = dash_duo.find_elements('.dash-cell')[1]
    cell.click()
    dash_duo.clear_input(cell)
    cell.send_keys('X')
    # save
    dash_duo.find_element('#save-patterns').click()
    dash_duo.wait_for_text_to_equal('#save-output','Saved 1 patterns.')
