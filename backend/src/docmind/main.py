"""ASGI entry point: ``uvicorn docmind.main:app``."""

from docmind.api.app import create_app

app = create_app()
