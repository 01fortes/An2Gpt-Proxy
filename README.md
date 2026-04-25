# atoo — Anthropic→OpenAI proxy

Tiny FastAPI server that accepts requests in **Anthropic Messages API** format
(`POST /v1/messages`) and forwards them to any **OpenAI-compatible**
`/v1/chat/completions` endpoint. Made for pointing Claude Code at
Hugging Face Router, OpenRouter, vLLM, Ollama, etc.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python proxy.py \
  --port 8787 \
  --base-url https://router.huggingface.co/v1 \
  --api-key "$HF_TOKEN"
```

Optional flags:

- `--host 0.0.0.0` — bind address (default `127.0.0.1`)
- `--model google/gemma-2-27b-it:novita` — pin a model regardless of what
  the client requests (Claude Code will send `claude-*` model IDs which
  the upstream won't recognise, so this is usually what you want)

## Point Claude Code at it

```bash
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_AUTH_TOKEN=anything   # ignored by the proxy
claude
```

If you used `--model`, that overrides whatever Claude Code asks for.
Otherwise set the model via Claude Code's own model selection.

## What it translates

- Messages, system prompt, multi-turn history
- Multimodal user content: text + `image` blocks (base64 and URL sources)
  → OpenAI `image_url` parts
- Tool definitions (`tools` + `input_schema` → `tools` + `function.parameters`)
- Tool calls (`tool_use` block ↔ `tool_calls`)
- Tool results (`tool_result` block → `role: "tool"` message)
- `tool_choice` (`auto` / `any` / specific tool / `none`)
- Streaming SSE: synthesises Anthropic's
  `message_start` / `content_block_*` / `message_delta` / `message_stop`
  events from OpenAI's chunk stream
- `stop_reason` mapping (`stop`→`end_turn`, `length`→`max_tokens`,
  `tool_calls`→`tool_use`)
- Usage tokens (`prompt_tokens`/`completion_tokens` →
  `input_tokens`/`output_tokens`)

## What it does NOT do

- No prompt-caching translation (Anthropic `cache_control` blocks are ignored)
- No fine-grained billing or rate-limit headers
- No retries — if the upstream errors, the error is surfaced
- Auth on the proxy itself: assumes `127.0.0.1` is safe; do not expose publicly
