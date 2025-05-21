import httpx
import logging
import json
from typing import AsyncGenerator, Dict, Any

from fastapi import HTTPException, Request
from starlette.responses import StreamingResponse

from app.core.config import settings
from app.models.openai_schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None

# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

async def get_http_client() -> httpx.AsyncClient:
    """Return a singleton httpx.AsyncClient instance."""
    global _http_client
    if _http_client is None:
        timeout = httpx.Timeout(settings.VLLM_REQUEST_TIMEOUT, connect=10)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        _http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    return _http_client


async def close_http_client() -> None:
    """Close shared HTTP client on application shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None

# ---------------------------------------------------------------------------
# Patch helpers – make Qwen/DeepSeek‑R1 reasoning output OpenAI‑compatible
# ---------------------------------------------------------------------------

def _patch_reasoning_in_message(msg: Dict[str, Any]) -> None:
    """Ensure `content` field is present and **remove** `reasoning_content`.

    • If `content` is missing/empty **and** `reasoning_content` exists → move it.
    • Regardless, drop the `reasoning_content` key so downstream clients never
      see it (matches OpenAI schema exactly).
    """
    if msg.get("reasoning_content") is None:
        return  # nothing to do

    if msg.get("content") in (None, ""):
        msg["content"] = msg["reasoning_content"]

    # Always delete the auxiliary key to stay schema‑compatible
    del msg["reasoning_content"]


def _patch_reasoning_in_message(msg: Dict[str, Any]) -> None:
    if msg.get("reasoning_content") is None:
        return
    if msg.get("content") in (None, ""):
        msg["content"] = msg["reasoning_content"]
    del msg["reasoning_content"]


def _patch_reasoning_in_json_bytes(body: bytes) -> bytes:
    try:
        data = json.loads(body)
        for choice in data.get("choices", []):
            _patch_reasoning_in_message(choice.get("message", {}))
        return json.dumps(data).encode()
    except Exception as exc:
        logger.warning("Reasoning JSON patch failed – forwarding raw body: %s", exc)
        return body


async def _stream_patch(gen: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    async for chunk in gen:
        try:
            for event in chunk.split(b"\n\n"):
                if not event.startswith(b"data:"):
                    yield event + b"\n\n"
                    continue
                payload = event[5:].strip()
                if payload == b"[DONE]":
                    yield event + b"\n\n"
                    continue
                try:
                    obj = json.loads(payload)
                except Exception:
                    yield event + b"\n\n"
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if "reasoning_content" in delta:
                    if "content" not in delta or delta.get("content") in (None, ""):
                        delta["content"] = delta.pop("reasoning_content")
                    else:
                        delta.pop("reasoning_content", None)
                obj["choices"][0]["delta"] = delta
                yield b"data: " + json.dumps(obj).encode() + b"\n\n"
        except Exception as exc:
            logger.warning("Reasoning SSE patch error – forwarding original chunk: %s", exc)
            yield chunk

# ---------------------------------------------------------------------------
# Backend response handler
# ---------------------------------------------------------------------------

async def _handle_backend_response(
    response: httpx.Response,
    is_streaming_request: bool,
) -> StreamingResponse:
    """Proxy vLLM responses and normalise reasoning fields."""
    excluded = {
        "content-length",
        "transfer-encoding",
        "connection",
        "content-encoding",   # 追加这一行
    }
    # ------------------------------------------------------- STREAMING branch
    if is_streaming_request:
        if response.status_code != 200:
            error_payload = await response.aread()
            await response.aclose()
            logger.error(
                "Backend streaming error %s: %s", response.status_code, error_payload.decode(errors="ignore")
            )
            raise HTTPException(status_code=response.status_code, detail=error_payload.decode(errors="ignore"))

        async def patched_stream() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in _stream_patch(response.aiter_bytes()):
                    yield chunk
            except httpx.StreamError as exc:
                logger.error("Streaming interrupted: %s", exc)
            finally:
                await response.aclose()

        safe_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded}
        return StreamingResponse(
            patched_stream(),
            media_type=response.headers.get("content-type", "text/event-stream"),
            status_code=response.status_code,
            headers=safe_headers,
        )

    # ---------------------------------------------------- NON‑STREAMING branch
    response_content = await response.aread()
    await response.aclose()

    safe_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded}

    if response.status_code >= 400:
        logger.error("Backend error %s: %s", response.status_code, response_content.decode(errors="ignore"))
        return StreamingResponse(
            iter([response_content]),
            media_type=response.headers.get("content-type", "application/json"),
            status_code=response.status_code,
            headers=safe_headers,
        )

    patched_body = _patch_reasoning_in_json_bytes(response_content)
    return StreamingResponse(
        iter([patched_body]),
        media_type=response.headers.get("content-type", "application/json"),
        status_code=response.status_code,
        headers=safe_headers,
    )

# ---------------------------------------------------------------------------
# Chat‑completion forwarder (forces enable_thinking=False)
# ---------------------------------------------------------------------------

async def forward_chat_completion_request_to_vllm(
    original_chat_request: ChatCompletionRequest,
) -> StreamingResponse:
    client = await get_http_client()
    target_url = f"{settings.VLLM_BASE_URL}{settings.VLLM_CHAT_COMPLETIONS_ENDPOINT}"

    payload: Dict[str, Any] = original_chat_request.model_dump(exclude_unset=True)
    payload["chat_template_kwargs"] = {"enable_thinking": False}

    payload.setdefault("temperature", settings.DEFAULT_TEMPERATURE)
    payload.setdefault("top_p", settings.DEFAULT_TOP_P)
    payload.setdefault("top_k", settings.DEFAULT_TOP_K)
    payload.setdefault("presence_penalty", settings.DEFAULT_PRESENCE_PENALTY)
    payload.setdefault("model", original_chat_request.model)

    is_stream = bool(payload.get("stream"))

    try:
        req = client.build_request("POST", target_url, json=payload)
        resp = await client.send(req, stream=is_stream)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gateway timeout: backend LLM did not respond in time.")
    except (httpx.ConnectError, httpx.RequestError) as exc:
        logger.error("Cannot reach backend: %s", exc)
        raise HTTPException(status_code=503, detail="Backend LLM service unreachable.")

    return await _handle_backend_response(resp, is_stream)

# ---------------------------------------------------------------------------
# Generic proxy endpoint – forwards everything else unchanged
# ---------------------------------------------------------------------------

async def forward_generic_request_to_vllm(request: Request) -> StreamingResponse:
    client = await get_http_client()

    path_and_query = request.url.path + ("?" + request.url.query if request.url.query else "")
    target_url = f"{settings.VLLM_BASE_URL}{path_and_query}"

    excluded = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}

    body = await request.body()

    expect_stream = "text/event-stream" in request.headers.get("accept", "").lower()

    try:
        req = client.build_request(request.method, target_url, headers=headers, content=body or None)
        resp = await client.send(req, stream=True)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gateway timeout: backend LLM did not respond in time.")
    except (httpx.ConnectError, httpx.RequestError) as exc:
        logger.error("Cannot reach backend: %s", exc)
        raise HTTPException(status_code=503, detail="Backend LLM service unreachable.")

    return await _handle_backend_response(resp, expect_stream)
