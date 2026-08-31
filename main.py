"""
Application entrypoint. Equivalent to Application.java in Spring Boot —
creates the app and wires in the routers. No business logic here.

Run with: uvicorn main:app --reload
"""

import logging

from fastapi import FastAPI

from config import LOGGER_NAME
from routers.ask import router as ask_router

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(LOGGER_NAME)

app = FastAPI(title="PromptOps")

app.include_router(ask_router)