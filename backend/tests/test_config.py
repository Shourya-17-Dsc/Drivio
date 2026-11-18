from app.config import _csv


def test_cors_origins_parser_removes_whitespace_and_empty_values(monkeypatch):
    monkeypatch.setenv("DRIVIO_TEST_ORIGINS", " https://demo.example , ,http://localhost:5173 ")
    assert _csv("DRIVIO_TEST_ORIGINS", "fallback") == ("https://demo.example", "http://localhost:5173")
