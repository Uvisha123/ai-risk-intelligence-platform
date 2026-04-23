"""
FastAPI entrypoint with logging middleware and global exception handlers.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import RiskIntelligenceError, to_http_payload
from app.core.logger import app_logger
from app.db.database import init_db


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log inbound requests and status/timing for each response."""

    async def dispatch(self, request: Request, call_next: Callable):
        """Capture POST JSON bodies (truncated) and downstream status codes."""
        settings = get_settings()
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)

        req_body_txt = ""
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request = Request(request.scope, receive)
            try:
                parsed = json.loads(body.decode("utf-8"))
                req_body_txt = json.dumps(parsed)[:4000]
            except Exception:
                req_body_txt = body.decode("utf-8", errors="ignore")[:4000]

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        app_logger.info(
            "%s %s status=%s duration_ms=%s request_body=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            req_body_txt,
        )
        app_logger.info(
            "response_meta path=%s status=%s media_type=%s",
            request.url.path,
            response.status_code,
            getattr(response, "media_type", ""),
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Initialize database schema on startup."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Application factory wired with routes and middleware."""

    settings = get_settings()

    application = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestLoggingMiddleware)

    @application.exception_handler(RiskIntelligenceError)
    async def domain_error_handler(request: Request, exc: RiskIntelligenceError):  # noqa: ARG001
        """Normalize domain-layer failures to HTTP 400."""
        payload = to_http_payload(exc)
        app_logger.warning("domain_error code=%s message=%s", exc.code, exc.message)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload)

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):  # noqa: ARG001
        """Expose validation failures without leaking internals."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"code": "VALIDATION_ERROR", "detail": exc.errors()},
        )

    @application.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):  # noqa: ARG001
        """Catch-all safeguard logging stack traces server-side."""
        app_logger.exception("unhandled_error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"code": "INTERNAL_ERROR", "message": "Unexpected server error"},
        )

    @application.get("/health")
    async def health():
        """Kubernetes-style liveness probe."""
        return {"status": "ok"}

    application.include_router(api_router, prefix=settings.api_v1_prefix)

    return application


app = create_app()
