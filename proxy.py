"""Anthropic-to-OpenAI proxy.

Accepts requests in Anthropic Messages API format on a local port and
forwards them to any OpenAI-compatible /v1/chat/completions endpoint
(Hugging Face router, OpenRouter, vLLM, Ollama, etc.).

Usage:
    python proxy.py --port 8787 \
        --base-url https://router.huggingface.co/v1 \
        --api-key $HF_TOKEN

Then point Claude Code at it:
    export ANTHROPIC_BASE_URL=http://localhost:8787
    export ANTHROPIC_AUTH_TOKEN=anything
    export ANTHROPIC_MODEL="google/gemma-2-27b-it:novita"
    claude
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI request conversion
# ---------------------------------------------------------------------------

def _anthropic_content_to_openai(content: Any) -> Any:
    """Convert an Anthropic message `content` field into OpenAI form.

    Anthropic content can be a string OR a list of blocks:
      {type: "text", text: ...}
      {type: "image", source: {type: "base64", media_type: ..., data: ...}}
      {type: "image", source: {type: "url", url: ...}}
      {type: "tool_use", id, name, input}
      {type: "tool_result", tool_use_id, content, is_error?}
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            out.append({"type": "text", "text": str(block)})
            continue
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source", {})
            stype = src.get("type")
            if stype == "base64":
                media_type = src.get("media_type", "image/jpeg")
                data = src.get("data", "")
                url = f"data:{media_type};base64,{data}"
            elif stype == "url":
                url = src.get("url", "")
            else:
                continue
            out.append({"type": "image_url", "image_url": {"url": url}})
        elif btype == "input_image":
            # Some clients send OpenAI-style blocks already.
            out.append(block)
        elif btype == "tool_use":
            # tool_use is folded into the assistant message at a higher level,
            # not into `content`. Skip here; handled in `_convert_messages`.
            continue
        elif btype == "tool_result":
            # tool_result is split out into its own role=tool message; skip.
            continue
        else:
            # Unknown blocks: stringify so we don't drop info.
            out.append({"type": "text", "text": json.dumps(block)})
    # If only one text block, collapse to a plain string for max compatibility.
    if len(out) == 1 and out[0].get("type") == "text":
        return out[0]["text"]
    return out


def _convert_messages(
    messages: list[dict[str, Any]],
    system: Any,
) -> list[dict[str, Any]]:
    """Convert Anthropic messages + top-level system into OpenAI messages."""
    oai: list[dict[str, Any]] = []

    # System: Anthropic puts it at top level, possibly as a list of blocks.
    if system:
        if isinstance(system, str):
            sys_text = system
        elif isinstance(system, list):
            parts: list[str] = []
            for b in system:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif isinstance(b, str):
                    parts.append(b)
            sys_text = "\n\n".join(p for p in parts if p)
        else:
            sys_text = str(system)
        if sys_text:
            oai.append({"role": "system", "content": sys_text})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            # Pull out any tool_result blocks into separate role=tool messages.
            tool_results: list[dict[str, Any]] = []
            user_blocks: list[dict[str, Any]] | str
            if isinstance(content, list):
                remaining: list[dict[str, Any]] = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tr_content = b.get("content", "")
                        if isinstance(tr_content, list):
                            text_parts = [
                                x.get("text", "")
                                for x in tr_content
                                if isinstance(x, dict) and x.get("type") == "text"
                            ]
                            tr_text = "\n".join(text_parts)
                        else:
                            tr_text = str(tr_content)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id", ""),
                            "content": tr_text,
                        })
                    else:
                        remaining.append(b)
                user_blocks = _anthropic_content_to_openai(remaining) if remaining else ""
            else:
                user_blocks = content

            # Tool results must precede any user text in OpenAI's expected order.
            oai.extend(tool_results)
            if user_blocks:
                oai.append({"role": "user", "content": user_blocks})

        elif role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        tool_calls.append({
                            "id": b.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        })
            else:
                text_parts.append(str(content))

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            text = "".join(text_parts)
            assistant_msg["content"] = text if text else None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            oai.append(assistant_msg)
        else:
            # Pass-through for any unexpected role.
            oai.append({"role": role, "content": _anthropic_content_to_openai(content)})

    return oai


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def _convert_tool_choice(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, dict):
        ttype = tc.get("type")
        if ttype == "auto":
            return "auto"
        if ttype == "any":
            return "required"
        if ttype == "tool":
            return {"type": "function", "function": {"name": tc.get("name", "")}}
        if ttype == "none":
            return "none"
    return tc


def anthropic_to_openai_request(
    body: dict[str, Any],
    *,
    stream: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": _convert_messages(body.get("messages", []), body.get("system")),
        "stream": stream,
    }
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]
    tools = _convert_tools(body.get("tools"))
    if tools:
        out["tools"] = tools
    tc = _convert_tool_choice(body.get("tool_choice"))
    if tc is not None:
        out["tool_choice"] = tc
    if stream:
        out["stream_options"] = {"include_usage": True}
    return out


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic response conversion (non-streaming)
# ---------------------------------------------------------------------------

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def openai_to_anthropic_response(
    oai: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    choice = (oai.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    finish = choice.get("finish_reason") or "stop"

    content_blocks: list[dict[str, Any]] = []
    text = msg.get("content")
    if isinstance(text, list):
        # Some providers return list-of-parts.
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
            "name": fn.get("name", ""),
            "input": args,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = oai.get("usage") or {}
    return {
        "id": oai.get("id") or f"msg_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic SSE conversion (streaming)
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def stream_openai_to_anthropic(
    upstream: AsyncIterator[bytes],
    *,
    model: str,
) -> AsyncIterator[bytes]:
    """Translate an OpenAI SSE stream into Anthropic SSE events.

    Anthropic stream shape:
      message_start -> [content_block_start, content_block_delta..., content_block_stop]+
        -> message_delta(stop_reason, usage) -> message_stop
    Each content block has its own index. Text and tool_use are separate blocks.
    """
    message_id = f"msg_{uuid.uuid4().hex[:16]}"

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    text_index: int | None = None
    text_started = False

    # Index per OpenAI tool_call index -> Anthropic block index.
    tool_block_index: dict[int, int] = {}
    tool_started: dict[int, bool] = {}
    next_block_index = 0

    finish_reason: str | None = None
    usage: dict[str, Any] = {}

    buffer = b""
    async for chunk in upstream:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if data.get("usage"):
                usage = data["usage"]

            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            # Text delta
            content_piece = delta.get("content")
            if content_piece:
                if isinstance(content_piece, list):
                    content_piece = "".join(
                        p.get("text", "") for p in content_piece if isinstance(p, dict)
                    )
                if not text_started:
                    text_index = next_block_index
                    next_block_index += 1
                    text_started = True
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_index,
                    "delta": {"type": "text_delta", "text": content_piece},
                })

            # Tool call deltas
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                fn = tc.get("function") or {}
                if idx not in tool_block_index:
                    block_idx = next_block_index
                    next_block_index += 1
                    tool_block_index[idx] = block_idx
                    tool_started[idx] = True
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                            "name": fn.get("name", "") or "",
                            "input": {},
                        },
                    })
                args_piece = fn.get("arguments")
                if args_piece:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": tool_block_index[idx],
                        "delta": {"type": "input_json_delta", "partial_json": args_piece},
                    })

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    # Close any open blocks.
    if text_started and text_index is not None:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": text_index})
    for idx, started in tool_started.items():
        if started:
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": tool_block_index[idx],
            })

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": _STOP_REASON_MAP.get(finish_reason or "stop", "end_turn"),
            "stop_sequence": None,
        },
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    })
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(*, base_url: str, api_key: str, model_override: str | None = None) -> FastAPI:
    app = FastAPI(title="Anthropic→OpenAI proxy")
    base_url = base_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "upstream": base_url}

    @app.post("/v1/messages")
    async def messages(req: Request) -> Any:
        try:
            body = await req.json()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

        if model_override:
            body["model"] = model_override
        model = body.get("model", "")
        stream = bool(body.get("stream"))
        upstream_body = anthropic_to_openai_request(body, stream=stream)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

        if not stream:
            try:
                resp = await client.post(chat_url, json=upstream_body, headers=headers)
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc
            if resp.status_code >= 400:
                return JSONResponse(
                    status_code=resp.status_code,
                    content={
                        "type": "error",
                        "error": {
                            "type": "upstream_error",
                            "message": resp.text,
                        },
                    },
                )
            data = resp.json()
            return JSONResponse(openai_to_anthropic_response(data, model=model))

        # Streaming
        async def event_stream() -> AsyncIterator[bytes]:
            try:
                async with client.stream(
                    "POST", chat_url, json=upstream_body, headers=headers
                ) as r:
                    if r.status_code >= 400:
                        err_text = (await r.aread()).decode("utf-8", errors="replace")
                        yield _sse("error", {
                            "type": "error",
                            "error": {"type": "upstream_error", "message": err_text},
                        })
                        return
                    async for piece in stream_openai_to_anthropic(
                        r.aiter_bytes(), model=model
                    ):
                        yield piece
            except httpx.HTTPError as exc:
                yield _sse("error", {
                    "type": "error",
                    "error": {"type": "upstream_error", "message": str(exc)},
                })

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic→OpenAI proxy")
    parser.add_argument("--port", type=int, default=8787, help="local port to bind")
    parser.add_argument("--host", default="127.0.0.1", help="local host (default 127.0.0.1)")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. https://router.huggingface.co/v1",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="bearer token for the upstream service",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="optional: override the model on every request",
    )
    args = parser.parse_args()

    app = build_app(
        base_url=args.base_url,
        api_key=args.api_key,
        model_override=args.model,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
