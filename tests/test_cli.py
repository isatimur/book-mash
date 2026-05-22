from typer.testing import CliRunner

from book_mash.cli import app


runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "measure" in result.stdout
    assert "gate" in result.stdout
    assert "autoresearch" in result.stdout


def test_gate_prints_stub():
    result = runner.invoke(app, ["gate"])
    assert result.exit_code == 0
    assert "not yet" in result.stdout


def test_autoresearch_prints_stub():
    result = runner.invoke(app, ["autoresearch"])
    assert result.exit_code == 0
    assert "not yet" in result.stdout
