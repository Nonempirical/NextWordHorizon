"""
Main entry point for Next Word Horizon API.

This file exposes the FastAPI app instance for uvicorn to run.
It imports the app from api.server module.
"""

from api.server import app

# Export the app for uvicorn
__all__ = ["app"]

