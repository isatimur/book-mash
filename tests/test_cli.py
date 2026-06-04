from typer.testing import CliRunner

from book_mash.cli import app


runner = CliRunner()


def test_help_renders():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "config" in result.stdout.lower()


def test_help_mentions_measurement():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "measure" in result.stdout.lower()


def test_missing_config_fails_cleanly():
    result = runner.invoke(app, [])
    assert result.exit_code != 0
