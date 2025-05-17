import sys, pathlib

# Ensure local 'wave' package is imported, not stdlib
project_root = pathlib.Path(__file__).parent.parent.resolve().as_posix()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from typer.testing import CliRunner
from wave import app

runner = CliRunner()


def test_help():
    "Test main CLI help output"
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_ingest_help():
    "Test ingest CLI help"
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "CSV → Parquet ingestion" in result.stdout
    # command usage displayed correctly


def test_scan_help():
    "Test scan CLI help"
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "Compute matrix profile" in result.stdout
    # jobs option available in nested command help


def test_cluster_help():
    "Test cluster CLI help"
    result = runner.invoke(app, ["cluster", "--help"])
    assert result.exit_code == 0
    assert "Cluster motifs and build pattern catalog" in result.stdout


def test_ui_help():
    "Test ui CLI help"
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "Launch Dash UI" in result.stdout


def test_api_help():
    "Test api CLI help"
    result = runner.invoke(app, ["api", "--help"])
    assert result.exit_code == 0
    assert "Launch Pattern API" in result.stdout


def test_seer_help():
    "Test seer CLI help"
    result = runner.invoke(app, ["seer", "--help"])
    assert result.exit_code == 0
    assert "Streaming SeerAgent" in result.stdout
