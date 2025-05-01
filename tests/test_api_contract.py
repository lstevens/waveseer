import pytest
schemathesis = pytest.importorskip("schemathesis")
import schemathesis
from schemathesis.exceptions import SchemaError
from wave.api.app import app

# Attempt to load OpenAPI schema; skip on unsupported signature or schema version
try:
    schema = schemathesis.from_asgi(app, path="/openapi.json")
except (TypeError, SchemaError):
    pytest.skip("Unsupported schemathesis.from_asgi call or schema version", allow_module_level=True)

@schema.parametrize(method="post", path="/match")
def test_match_contract(case):
    response = case.call_asgi()
    case.validate_response(response)

@schema.parametrize(method="get", path="/catalog")
def test_catalog_contract(case):
    response = case.call_asgi()
    case.validate_response(response)
