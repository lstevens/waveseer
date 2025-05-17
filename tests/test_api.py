import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup ML mocks if in testing mode
import os
from unittest.mock import MagicMock
from wave.test_utils.ml_mocks import setup_ml_mocks
from wave.test_utils.decorators import requires_torch, requires_ml_stack

is_testing = setup_ml_mocks()

# Import torch after mocks have been set up
import torch

import numpy as np
from fastapi.testclient import TestClient
from wave.api.app import app, MODELS_DIR

# Dummy DB for testing
class DummyDB:
    def execute(self, query, *args, **kwargs):
        if "SELECT pattern_id FROM patterns WHERE label LIKE" in query:  # ML match endpoint
            class Res:
                def fetchall(self):
                    return [("ml_pattern_123",)]
            return Res()
        if query.strip().startswith("SELECT pattern_id FROM patterns"):  # match endpoint
            class Res:
                def fetchall(self):
                    return [("pid123",)]
            return Res()
        if query.strip().startswith("SELECT pattern_id, template, label, color"):  # match with template
            class DF:
                def df(self):
                    import pandas as pd
                    return pd.DataFrame([{
                        "pattern_id": "pid123", 
                        "template": "[0.1, 0.2, 0.3, 0.4, 0.5]",
                        "label": "test_pattern", 
                        "color": "blue"
                    }])
            return DF()
        if query.strip().startswith("SELECT pattern_id, label, color"):  # catalog with color
            class DF:
                def df(self):
                    import pandas as pd
                    return pd.DataFrame([{"pattern_id": "pid123", "label": "L", "color": "C"}])
            return DF()
        if query.strip().startswith("SELECT pattern_id, label"):  # fallback catalog
            class DF:
                def df(self):
                    import pandas as pd
                    return pd.DataFrame([{"pattern_id": "pid123", "label": "L", "color": ""}])
            return DF()
        if query.strip().startswith("UPDATE patterns SET"):  # update_pattern
            return None
        return self

# Mock TorchScript model
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Return a tensor with confidence 0.85
        return torch.tensor([[0.85]])

# Mock class and function for testing
def mock_load_exported_model(path):
    return MockModel()

def mock_calculate_pattern_similarity(seq1, seq2):
    return 0.75  # Fixed similarity score for testing

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch database
    import wave.api.app as api_app
    monkeypatch.setattr(api_app, 'duckdb', type('m', (), {'connect': lambda _: DummyDB()}))
    
    # Patch model loading
    monkeypatch.setattr(api_app, 'load_exported_model', mock_load_exported_model)
    monkeypatch.setattr(api_app, 'calculate_pattern_similarity', mock_calculate_pattern_similarity)
    
    # Create mock model directory and files for testing
    mock_models_dir = Path("/tmp/waveseer_test_models")
    mock_models_dir.mkdir(exist_ok=True)
    
    # Create a dummy model file
    dummy_model = mock_models_dir / "test_model.pt"
    dummy_model.write_bytes(b"dummy model content")
    
    # Create a dummy config file
    dummy_config = mock_models_dir / "test_model_config.json"
    config_content = {
        "model_type": "cnn",
        "classes": {"0": "no_pattern", "1": "head_shoulders", "2": "double_top"},
        "metadata": {"accuracy": 0.92, "f1_score": 0.89}
    }
    dummy_config.write_text(json.dumps(config_content))
    
    # Patch the models directory
    monkeypatch.setattr(api_app, 'MODELS_DIR', mock_models_dir)

client = TestClient(app)

# Original API tests
@requires_torch
def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

@requires_torch
def test_traditional_match():
    """Test traditional rule-based matching"""
    payload = {"tf": "1m", "seq": [1.0, 2.0, 3.0, 4.0, 5.0], "use_ml": False}
    r = client.post('/match', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['pattern_id'] == 'pid123'
    assert isinstance(data['score'], float)
    assert isinstance(data['dist'], float)
    assert 'detection_time_ms' in data

@requires_torch
def test_ml_match():
    """Test ML-based pattern matching"""
    payload = {
        "tf": "1m", 
        "seq": [1.0, 2.0, 3.0, 4.0, 5.0], 
        "use_ml": True,
        "model_name": "test_model",
        "confidence_threshold": 0.5
    }
    r = client.post('/match', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['pattern_id'] == 'ml_pattern_123'
    assert data['score'] >= 0.7  # The model confidence after processing
    assert data['ml_model'] == 'test_model'
    assert 'detection_time_ms' in data

@requires_torch
def test_catalog():
    r = client.get('/catalog')
    assert r.status_code == 200
    json = r.json()
    assert 'patterns' in json
    patterns = json['patterns']
    assert patterns[0]['pattern_id'] == 'pid123'
    assert patterns[0]['label'] == 'L'

@pytest.mark.parametrize("pid", ["pidA", "pidB"])
@requires_torch
def test_update_pattern(pid):
    r = client.put(f'/patterns/{pid}', json={"label": "new", "color": "blue"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

# New API endpoint tests
@requires_torch
def test_list_models():
    """Test listing available models"""
    r = client.get('/models')
    assert r.status_code == 200
    models = r.json()
    assert len(models) >= 1
    assert any(model['name'] == 'test_model' for model in models)
    
    # Check model details
    model = next(m for m in models if m['name'] == 'test_model')
    assert model['type'] == 'cnn'
    assert model['has_config'] == True
    assert len(model['classes']) == 3
    assert 'head_shoulders' in model['classes']

@requires_torch
def test_get_model_info():
    """Test getting specific model info"""
    r = client.get('/models/test_model')
    assert r.status_code == 200
    model = r.json()
    assert model['name'] == 'test_model'
    assert model['type'] == 'cnn'
    assert model['has_config'] == True
    assert 'size_bytes' in model
    
    # Verify metadata is present
    assert 'metadata' in model
    assert model['metadata']['accuracy'] == 0.92

@requires_torch
def test_nonexistent_model():
    """Test requesting a model that doesn't exist"""
    r = client.get('/models/nonexistent_model')
    assert r.status_code == 404

@requires_torch
def test_batch_match():
    """Test batch processing endpoint"""
    payload = {
        "sequences": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0]
        ],
        "tf": "1h",
        "use_ml": True,
        "model_name": "test_model"
    }
    r = client.post('/batch/match', json=payload)
    assert r.status_code == 200
    data = r.json()
    
    # Check response structure
    assert 'results' in data
    assert 'total_time_ms' in data
    assert 'avg_time_ms' in data
    
    # Check results
    assert len(data['results']) == 2
    assert all(result['ml_model'] == 'test_model' for result in data['results'])

@requires_torch
def test_test_model_endpoint():
    """Test the model testing endpoint"""
    r = client.get('/test-model/test_model?sequence_length=50')
    assert r.status_code == 200
    data = r.json()
    
    assert data['test_status'] == 'success'
    assert data['model_name'] == 'test_model'
    assert 'inference_result' in data
    assert data['sequence_length'] == 50

@requires_torch
def test_root_endpoint():
    """Test the root endpoint"""
    r = client.get('/')
    assert r.status_code == 200
    data = r.json()
    
    assert 'name' in data
    assert 'version' in data
    assert 'documentation' in data
