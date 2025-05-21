import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time

from app.core.config import settings, get_settings
from app.core.logging_config import setup_logging
from app.models.openai_schemas import ChatCompletionRequest # For the specific endpoint
from app.services.vllm_service import (
    forward_chat_completion_request_to_vllm, # Renamed from forward_request_to_vllm
    forward_generic_request_to_vllm,         # New function
    close_http_client
)

get_settings()
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing resources.")
    # httpx client is initialized lazily
    yield
    logger.info("Application shutdown: Cleaning up resources.")
    await close_http_client()

app = FastAPI(
    title="Qwen3 nothink API Gateway",
    description="An API gateway to forward requests to a vLLM backend, enabling Qwen's non-thinking mode for chat completions and proxying other requests.",
    version="0.2.0", # Version bump
    lifespan=lifespan
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Response: {response.status_code} - "
            f"Duration: {process_time:.4f}s"
        )
        return response
app.add_middleware(RequestLoggingMiddleware)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url.path}: {exc}", exc_info=True)
    # Avoid returning HTTPException directly here if it's already one
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred."},
    )

# --- API Endpoints ---

# 1. Specific route for Chat Completions (Qwen non-thinking mode applied)
@app.post(settings.VLLM_CHAT_COMPLETIONS_ENDPOINT, summary="Forward Chat Completion Request to Qwen-vLLM with non-thinking mode")
async def chat_completions_custom(
    request_body: ChatCompletionRequest # Use the Pydantic model for request body validation
):
    """
    Receives an OpenAI-compatible chat completion request, modifies it to enable
    Qwen's non-thinking mode, and forwards it to the vLLM backend server.
    """
    try:
        return await forward_chat_completion_request_to_vllm(request_body)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing chat completion request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# 2. Generic proxy for all other routes (e.g., /v1/models, /v1/completions (non-chat), etc.)
# This route MUST be defined AFTER any specific routes like the one above.
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def generic_proxy(request: Request): # Removed full_path from args as request.url.path is better
    """
    A generic proxy that forwards requests to the vLLM backend as-is.
    Handles paths like /v1/models, /v1/completions, etc.
    """
    logger.info(f"Generic proxy handling request for: {request.method} {request.url.path}")
    try:
        return await forward_generic_request_to_vllm(request)
    except HTTPException as e: # Re-raise HTTPException to let FastAPI handle it
        raise e
    except Exception as e: # Catch any other unexpected errors from the service layer
        logger.error(f"Error in generic proxy for {request.url.path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during generic proxy: {str(e)}")


@app.get("/gateway/health", summary="Gateway Health Check", tags=["Gateway Management"])
async def health_check():
    return {"status": "ok", "service": "Qwen API Gateway"}

if __name__ == "__main__": # Keep for direct execution if needed
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.LOG_LEVEL.lower())