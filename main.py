# free-model-aggregator-02: one server that puts OpenRouter's free models
# behind a single endpoint, speaking both the OpenAI and the Anthropic wire.
#
# The server reads the trial brief (MAKEFUN_PROMPT_FILE), registers its port
# and catalog with the env (MAKEFUN_ENV_URL), and dials the brief's free
# models through the grader's gateway, always on Chat Completions. Every
# choice is a function of the brief, the requests and the gateway's answers:
# no wall clock and no unseeded randomness ever reaches a request.
#
# Run:   python3 main.py            (trial mode: register, serve, exit 0)
#        python3 main.py --list     (print the catalog document, exit 0)
#
# Python 3.13, standard library only.

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OPENROUTER_PUBLIC_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
SCHEMA_PREFIX = "free-model-aggregator-02/"
MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1"
MAX_TOKENS_DEFAULT = 1024
MAX_TOKENS_CAP = 8192
GATEWAY_TIMEOUT_SECONDS = 25
MAX_GATEWAY_BYTES = 8 * 1024 * 1024
MAX_TRAIL_ROWS = 8
MAX_CANDIDATES_PER_PROMPT = 3
RETRY_PAUSE_SECONDS = 2
WORK_POLL_SECONDS = 0.5


def _object_end(text, start):
    """Index of the '}' closing the object that opens at start, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in text that carries a string schema, in order."""
    found = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                found.append(value)
        start = text.find("{", start + 1)
    return found


def _string_of(value):
    return value if isinstance(value, str) else ""


def _number_or_none(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
        return value
    return None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


def _array_document(documents, keys):
    """The first document under this contest's prefix holding one of keys."""
    for doc in documents:
        if SCHEMA_PREFIX not in _string_of(doc.get("schema")):
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None, None


def _model_of(value):
    """One free-model row of the brief, keyed by the id the gateway routes."""
    if not isinstance(value, dict):
        return None
    raw = _string_of(value.get("id")) or _string_of(value.get("model"))
    if not raw:
        return None
    named = _string_of(value.get("provider")) or _string_of(value.get("wire"))
    if named == "openrouter" and not raw.lower().startswith("openrouter/"):
        model_id = "openrouter/" + raw
    else:
        model_id = raw
    return {
        "id": model_id,
        "provider": named or "openrouter",
        "display_name": _string_of(value.get("display_name")) or raw,
        "rating_milli": _number_or_none(value.get("rating_milli")) or 0,
        "context_length": _number_or_none(value.get("context_length")),
        "tools": _bool_or_none(value.get("tools")),
        "vision": _bool_or_none(value.get("vision")),
        "json": _bool_or_none(value.get("json")),
        "free": value.get("free") is not False,
    }


def _prompt_of(value):
    """One prompt row of the brief's manifest."""
    if not isinstance(value, dict):
        return None
    prompt_id = _string_of(value.get("id"))
    if not prompt_id:
        return None
    requires = value.get("requires") if isinstance(value.get("requires"), dict) else {}
    clean = {}
    for key in ("tools", "vision", "json"):
        if requires.get(key) is True:
            clean[key] = True
    min_context = _number_or_none(requires.get("min_context"))
    if min_context is not None:
        clean["min_context"] = min_context
    return {
        "id": prompt_id,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": clean,
    }


def read_brief(path):
    """The brief at path: its free model list and its prompt manifest."""
    if not path:
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return {"models": [], "prompts": [], "error": "cannot read %s: %s" % (path, exc)}
    documents = _documents_in(text)
    models_doc, models_key = _array_document(documents, ("models", "ratings"))
    if models_doc is None:
        return {"models": [], "prompts": [],
                "error": "the brief at %s holds no %s document" % (path, MODELS_SCHEMA)}
    models = []
    for row in models_doc[models_key]:
        model = _model_of(row)
        if model is not None:
            models.append(model)
    prompts_doc, prompts_key = _array_document(documents, ("prompts", "manifest"))
    if prompts_doc is None:
        return {"models": models, "prompts": [],
                "error": "the brief at %s holds no %s document" % (path, PROMPTS_SCHEMA)}
    prompts = []
    for row in prompts_doc[prompts_key]:
        prompt = _prompt_of(row)
        if prompt is not None:
            prompts.append(prompt)
    return {"models": models, "prompts": prompts, "error": None}


def build_catalog(brief):
    """The catalog document: one provider, the brief's free models."""
    seen = set()
    models = []
    for model in brief["models"]:
        if not model["free"] or model["id"] in seen:
            continue
        seen.add(model["id"])
        models.append({
            "id": model["id"],
            "free": True,
            "context_length": model["context_length"],
        })
    if not models:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": models,
            }
        ]
    }


def catalog_text(brief):
    """The catalog document as one compact JSON string, byte-stable."""
    return json.dumps(build_catalog(brief), separators=(",", ":"))


def bare_model_id(dialable_id):
    """The id a completion echoes: the dialable id without openrouter/."""
    if dialable_id.startswith("openrouter/"):
        return dialable_id[len("openrouter/"):]
    return dialable_id


def meets_requirements(requires, model):
    """Whether a model's capabilities meet one prompt's requirements."""
    if requires.get("tools") is True and model["tools"] is not True:
        return False
    if requires.get("vision") is True and model["vision"] is not True:
        return False
    if requires.get("json") is True and model["json"] is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        context = model["context_length"]
        if context is None or context < min_context:
            return False
    return True


def rank_candidates(models, requires):
    """Free models meeting the requirements, best first.

    Order: house rating, then context length, then the brief's own order.
    """
    ranked = []
    for index, model in enumerate(models):
        if not model["free"]:
            continue
        if not meets_requirements(requires, model):
            continue
        context = model["context_length"]
        ranked.append((-model["rating_milli"], -(context if context is not None else 0), index, model))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    return [row[3] for row in ranked]


def requirements_of_openai(body):
    """What an OpenAI-shaped request needs from its model."""
    requires = {}
    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") in ("json_object", "json_schema"):
        requires["json"] = True
    if body.get("tools"):
        requires["tools"] = True
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    requires["vision"] = True
    return requires


def requirements_of_anthropic(body):
    """What an Anthropic-shaped request needs from its model."""
    requires = {}
    if body.get("tools"):
        requires["tools"] = True
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    requires["vision"] = True
    return requires


def capped_max_tokens(body):
    """The request's output cap: default 1024, never over 8192."""
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return MAX_TOKENS_DEFAULT
    value = int(raw)
    if value <= 0:
        return MAX_TOKENS_DEFAULT
    return min(value, MAX_TOKENS_CAP)


def fallback_usage(text):
    """Usage when the provider sends none: input 0, output at least 1."""
    completion = max(1, len(text) // 4)
    return {"prompt_tokens": 0, "completion_tokens": completion, "total_tokens": completion}


STOP_REASON_BY_FINISH = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def anthropic_stop_reason(finish_reason):
    """A Chat Completions finish reason as an Anthropic stop reason."""
    return STOP_REASON_BY_FINISH.get(finish_reason, "end_turn")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise ValueError("redirect refused")


def _gateway_opener():
    return urllib.request.build_opener(_NoRedirect())


def call_gateway(cc_body, state):
    """POST one Chat Completions body to the gateway.

    Returns (status, headers, raw bytes). A transport failure is reported
    as a 502 with a gateway_error body, so the caller can fail over.
    """
    url = state["openai_base_url"].rstrip("/") + "/chat/completions"
    payload = json.dumps(cc_body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", "Bearer " + state["openai_api_key"])
    try:
        with _gateway_opener().open(request, timeout=GATEWAY_TIMEOUT_SECONDS) as response:
            return response.status, dict(response.headers), response.read(MAX_GATEWAY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(MAX_GATEWAY_BYTES + 1)
        except Exception:
            raw = b""
        return exc.code, dict(exc.headers), raw
    except Exception as exc:  # connection refused, timeout, redirect, ...
        error = {"error": {"message": str(exc), "type": "gateway_error", "code": "gateway_error"}}
        return 502, {}, json.dumps(error).encode("utf-8")


def parse_sse(raw):
    """SSE bytes -> (chunks, saw_done): parsed JSON data frames in order."""
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    chunks = []
    saw_done = False
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if not data_lines:
            continue
        data = "\n".join(data_lines).strip()
        if data == "[DONE]":
            saw_done = True
            continue
        try:
            chunks.append(json.loads(data))
        except ValueError:
            continue
    return chunks, saw_done


def error_code_of(raw):
    """The short error code of an upstream error body, or 'error'."""
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return "error"
    error = value.get("error") if isinstance(value, dict) else None
    if not isinstance(error, dict):
        return "error"
    code = error.get("code") or error.get("type") or "error"
    return code if isinstance(code, str) and code else "error"


def _text_of_content(content):
    """Plain text of a message content: a string, or its text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def anthropic_tool_to_cc(tool):
    """One Anthropic tool definition as an OpenAI function tool."""
    return {
        "type": "function",
        "function": {
            "name": tool.get("name") or "",
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object"},
        },
    }


def anthropic_tool_choice_to_cc(choice):
    """An Anthropic tool_choice as an OpenAI tool_choice."""
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"type": "function", "function": {"name": choice.get("name") or ""}}
    return "auto"


def anthropic_to_cc(body, model_id, max_tokens):
    """An Anthropic Messages body as a Chat Completions body."""
    messages = []
    system = body.get("system")
    if system is not None:
        text = _text_of_content(system)
        if text:
            messages.append({"role": "system", "content": text})
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        text_parts = []
        image_parts = []
        tool_calls = []
        if isinstance(content, str):
            text_parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append({"type": "text", "text": text})
                elif kind == "image":
                    source = part.get("source") if isinstance(part.get("source"), dict) else {}
                    if source.get("type") == "base64":
                        media = source.get("media_type") or "image/png"
                        data = source.get("data") or ""
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": "data:%s;base64,%s" % (media, data)},
                        })
                elif kind == "tool_use":
                    arguments = part.get("input")
                    tool_calls.append({
                        "id": part.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": part.get("name") or "",
                            "arguments": arguments if isinstance(arguments, str)
                                else json.dumps(arguments if arguments is not None else {}),
                        },
                    })
                elif kind == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": part.get("tool_use_id") or "",
                        "content": _text_of_content(part.get("content")),
                    })
        if role == "user":
            if text_parts or image_parts:
                messages.append({"role": "user", "content": text_parts + image_parts})
        elif role == "assistant":
            assistant = {"role": "assistant"}
            if text_parts:
                assistant["content"] = text_parts
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            if text_parts or tool_calls:
                messages.append(assistant)
    cc = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
    temperature = body.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        cc["temperature"] = temperature
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        cc["stop"] = stop
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        cc["tools"] = [anthropic_tool_to_cc(tool) for tool in tools if isinstance(tool, dict)]
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            cc["tool_choice"] = anthropic_tool_choice_to_cc(choice)
    return cc


def cc_to_anthropic(answer, bare_id):
    """A Chat Completions answer as an Anthropic Messages answer."""
    choice = {}
    choices = answer.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    finish = choice.get("finish_reason")
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except ValueError:
                    parsed = {}
            elif isinstance(arguments, dict):
                parsed = arguments
            else:
                parsed = {}
            content.append({
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": function.get("name") or "",
                "input": parsed,
            })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = answer.get("usage") if isinstance(answer.get("usage"), dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, (int, float)) or isinstance(prompt_tokens, bool):
        prompt_tokens = 0
    if not isinstance(completion_tokens, (int, float)) or isinstance(completion_tokens, bool):
        completion_tokens = max(1, len(_text_of_content(message.get("content"))) // 4)
    message_id = answer.get("id")
    if isinstance(message_id, str) and message_id.startswith("chatcmpl-"):
        message_id = "msg-" + message_id[len("chatcmpl-"):]
    elif not isinstance(message_id, str) or not message_id:
        message_id = "msg-aggregated"
    else:
        message_id = "msg-" + message_id
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": bare_id,
        "stop_reason": anthropic_stop_reason(finish),
        "stop_sequence": None,
        "usage": {"input_tokens": int(prompt_tokens), "output_tokens": int(completion_tokens)},
    }


def anthropic_stream_events(answer_chunks, bare_id):
    """Collected Chat Completions chunks as Anthropic SSE event texts."""
    text = ""
    tool_calls = {}
    finish = None
    usage = None
    message_id = "msg-aggregated"
    for chunk in answer_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id.startswith("chatcmpl-"):
            message_id = "msg-" + chunk_id[len("chatcmpl-"):]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        piece = delta.get("content")
        if isinstance(piece, str):
            text += piece
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                index = call.get("index")
                if not isinstance(index, int):
                    index = 0
                slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if isinstance(call.get("id"), str):
                    slot["id"] = call["id"]
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                if isinstance(function.get("name"), str):
                    slot["name"] = function["name"]
                fragment = function.get("arguments")
                if isinstance(fragment, str):
                    slot["arguments"] += fragment
        reason = choice.get("finish_reason")
        if isinstance(reason, str):
            finish = reason
    prompt_tokens = 0
    completion_tokens = None
    if isinstance(usage, dict):
        raw_in = usage.get("prompt_tokens")
        raw_out = usage.get("completion_tokens")
        if isinstance(raw_in, (int, float)) and not isinstance(raw_in, bool):
            prompt_tokens = int(raw_in)
        if isinstance(raw_out, (int, float)) and not isinstance(raw_out, bool):
            completion_tokens = int(raw_out)
    if completion_tokens is None:
        completion_tokens = max(1, len(text) // 4)
    stop = anthropic_stop_reason(finish)
    if tool_calls and finish is None:
        stop = "tool_use"
    events = []
    events.append(("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": bare_id,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        },
    }))
    block = 0
    if text:
        events.append(("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }))
        events.append(("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
        block = 1
    for index in sorted(tool_calls):
        call = tool_calls[index]
        try:
            parsed = json.loads(call["arguments"]) if call["arguments"] else {}
        except ValueError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        events.append(("content_block_start", {
            "type": "content_block_start", "index": block,
            "content_block": {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}},
        }))
        events.append(("content_block_delta", {
            "type": "content_block_delta", "index": block,
            "delta": {"type": "input_json_delta", "partial_json": call["arguments"]},
        }))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": block}))
        block += 1
    events.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": completion_tokens},
    }))
    events.append(("message_stop", {"type": "message_stop"}))
    return ["event: %s\ndata: %s\n\n" % (name, json.dumps(payload, separators=(",", ":")))
            for name, payload in events]


def build_cc_body(body, model_id, max_tokens, stream):
    """An OpenAI-shaped request as the gateway body for one model."""
    cc = {"model": model_id, "messages": body.get("messages") or [], "max_tokens": max_tokens}
    temperature = body.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        cc["temperature"] = temperature
    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        cc["response_format"] = response_format
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        cc["tools"] = tools
        choice = body.get("tool_choice")
        if choice is not None:
            cc["tool_choice"] = choice
    stop = body.get("stop")
    if stop is not None:
        cc["stop"] = stop
    if stream:
        cc["stream"] = True
        cc["stream_options"] = {"include_usage": True}
    return cc


def dial_with_failover(call, candidates, state):
    """Dial the gateway for one prompt, failing over on refusals.

    call(model_id) -> (status, headers, raw). Returns
    (dialled_model, status, headers, raw, failures): dialled_model is None
    when no call succeeded, and failures is the trail of refusals in order.
    A model that answers 429 is cooled down for the rest of the trial.
    """
    failures = []
    last = None
    for model in candidates[:MAX_CANDIDATES_PER_PROMPT]:
        if model["id"] in state["cooled"]:
            continue
        status, headers, raw = call(model["id"])
        last = (status, headers, raw)
        if 200 <= status < 300:
            return model, status, headers, raw, failures
        failures.append({"status": status, "code": error_code_of(raw), "model": model["id"]})
        if status == 429:
            state["cooled"].add(model["id"])
    if failures and candidates:
        time.sleep(RETRY_PAUSE_SECONDS)
        best = candidates[0]
        status, headers, raw = call(best["id"])
        last = (status, headers, raw)
        if 200 <= status < 300:
            return best, status, headers, raw, failures
        failures.append({"status": status, "code": error_code_of(raw), "model": best["id"]})
        if status == 429:
            state["cooled"].add(best["id"])
    if last is None:
        error = {"error": {"message": "no free model meets this prompt",
                           "type": "invalid_request_error", "code": "model_not_allowed"}}
        return None, 400, {}, json.dumps(error).encode("utf-8"), failures
    status, headers, raw = last
    return None, status, headers, raw, failures


def trail_header(failures):
    """The upstream-failures header value for a prompt, or None."""
    if not failures:
        return None
    return json.dumps(failures[:MAX_TRAIL_ROWS], separators=(",", ":"))


def _candidates_for(body, state, requires):
    """The models one request may dial: the named one, or the best free."""
    requested = body.get("model")
    if isinstance(requested, str) and requested not in ("", "auto"):
        named = [model for model in state["models"] if model["id"] == requested]
        if named:
            return named
        return []
    return rank_candidates(state["models"], requires)


def _no_candidate_headers():
    return {"Content-Type": "application/json"}


def _no_candidate_body():
    return json.dumps({"error": {"message": "no free model meets this prompt",
                                 "type": "invalid_request_error",
                                 "code": "model_not_allowed"}}).encode("utf-8")


def openai_stream_text(chunks, bare):
    """Upstream Chat Completions chunks as one SSE text for the client."""
    frames = []
    text = ""
    saw_finish = False
    saw_usage = False
    created = 0
    chunk_id = "chatcmpl-aggregated"
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk["model"] = bare
        if isinstance(chunk.get("created"), int):
            created = chunk["created"]
        if isinstance(chunk.get("id"), str) and chunk["id"]:
            chunk_id = chunk["id"]
        if isinstance(chunk.get("usage"), dict):
            saw_usage = True
        choices = chunk.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    text += delta["content"]
                if isinstance(choice.get("finish_reason"), str):
                    saw_finish = True
        frames.append("data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n")
    if not saw_finish:
        frames.append("data: " + json.dumps({
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": bare, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }, separators=(",", ":")) + "\n\n")
    if not saw_usage:
        frames.append("data: " + json.dumps({
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": bare, "choices": [],
            "usage": fallback_usage(text),
        }, separators=(",", ":")) + "\n\n")
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def serve_openai(body, state):
    """Answer an OpenAI Chat Completions request: (status, headers, payload)."""
    stream = body.get("stream") is True
    max_tokens = capped_max_tokens(body)
    candidates = _candidates_for(body, state, requirements_of_openai(body))
    if not candidates:
        return 400, _no_candidate_headers(), _no_candidate_body()

    def call(model_id):
        return call_gateway(build_cc_body(body, model_id, max_tokens, stream), state)

    model, status, headers, raw, failures = dial_with_failover(call, candidates, state)
    extra = {}
    trail = trail_header(failures)
    if trail is not None:
        extra["x-makefun-upstream-failures"] = trail
    if model is None:
        out_headers = {"Content-Type": "application/json"}
        out_headers.update(extra)
        return status, out_headers, raw
    bare = bare_model_id(model["id"])
    if stream:
        chunks, _saw_done = parse_sse(raw)
        out_headers = {"Content-Type": "text/event-stream"}
        out_headers.update(extra)
        return 200, out_headers, openai_stream_text(chunks, bare).encode("utf-8")
    try:
        answer = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        error = {"error": {"message": "the gateway answered a body that is not JSON",
                           "type": "gateway_error", "code": "gateway_error"}}
        return 502, {"Content-Type": "application/json"}, json.dumps(error).encode("utf-8")
    if not isinstance(answer, dict):
        error = {"error": {"message": "the gateway answered a JSON value that is not one object",
                           "type": "gateway_error", "code": "gateway_error"}}
        return 502, {"Content-Type": "application/json"}, json.dumps(error).encode("utf-8")
    answer["model"] = bare
    usage = answer.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    text = ""
    choices = answer.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            text = _text_of_content(message.get("content"))
        if choices[0].get("finish_reason") is None:
            choices[0]["finish_reason"] = "stop"
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, (int, float)) or isinstance(prompt_tokens, bool):
        prompt_tokens = 0
    if not isinstance(completion_tokens, (int, float)) or isinstance(completion_tokens, bool):
        completion_tokens = max(1, len(text) // 4)
    answer["usage"] = {"prompt_tokens": int(prompt_tokens),
                       "completion_tokens": int(completion_tokens),
                       "total_tokens": int(prompt_tokens) + int(completion_tokens)}
    out_headers = {"Content-Type": "application/json"}
    out_headers.update(extra)
    return 200, out_headers, json.dumps(answer, separators=(",", ":")).encode("utf-8")


def serve_anthropic(body, state):
    """Answer an Anthropic Messages request: (status, headers, payload)."""
    stream = body.get("stream") is True
    max_tokens = capped_max_tokens(body)
    candidates = _candidates_for(body, state, requirements_of_anthropic(body))
    if not candidates:
        return 400, _no_candidate_headers(), _no_candidate_body()

    def call(model_id):
        cc = anthropic_to_cc(body, model_id, max_tokens)
        if stream:
            cc["stream"] = True
            cc["stream_options"] = {"include_usage": True}
        return call_gateway(cc, state)

    model, status, headers, raw, failures = dial_with_failover(call, candidates, state)
    extra = {}
    trail = trail_header(failures)
    if trail is not None:
        extra["x-makefun-upstream-failures"] = trail
    if model is None:
        out_headers = {"Content-Type": "application/json"}
        out_headers.update(extra)
        return status, out_headers, raw
    bare = bare_model_id(model["id"])
    if stream:
        chunks, _saw_done = parse_sse(raw)
        events = anthropic_stream_events(chunks, bare)
        out_headers = {"Content-Type": "text/event-stream"}
        out_headers.update(extra)
        return 200, out_headers, "".join(events).encode("utf-8")
    try:
        answer = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        error = {"error": {"message": "the gateway answered a body that is not JSON",
                           "type": "gateway_error", "code": "gateway_error"}}
        return 502, {"Content-Type": "application/json"}, json.dumps(error).encode("utf-8")
    if not isinstance(answer, dict):
        error = {"error": {"message": "the gateway answered a JSON value that is not one object",
                           "type": "gateway_error", "code": "gateway_error"}}
        return 502, {"Content-Type": "application/json"}, json.dumps(error).encode("utf-8")
    message = cc_to_anthropic(answer, bare)
    out_headers = {"Content-Type": "application/json"}
    out_headers.update(extra)
    return 200, out_headers, json.dumps(message, separators=(",", ":")).encode("utf-8")


def models_openai_payload(models):
    """The /v1/models answer on the OpenAI wire: the dialable ids."""
    return {
        "object": "list",
        "data": [
            {"id": model["id"], "object": "model", "created": 0, "owned_by": "openrouter"}
            for model in models
        ],
    }


def models_anthropic_payload(models, brief):
    """The /v1/models answer on the Anthropic wire: the dialable ids."""
    names = {}
    for model in brief["models"]:
        names[model["id"]] = model["display_name"]
    data = [
        {
            "id": model["id"],
            "type": "model",
            "display_name": names.get(model["id"], model["id"]),
            "created_at": "2026-01-01T00:00:00Z",
        }
        for model in models
    ]
    payload = {"data": data, "has_more": False}
    if data:
        payload["first_id"] = data[0]["id"]
        payload["last_id"] = data[-1]["id"]
    return payload


class AggregatorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state = None  # set by main: {"brief", "models", "cooled", ...}

    def log_message(self, *args):
        return

    def _send(self, status, headers, payload):
        body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            size = 0
        raw = self.rfile.read(size) if size > 0 else b""
        try:
            value = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            return None, False
        return value, True

    def _error(self, status, code, message):
        error = {"error": {"message": message, "type": code, "code": code}}
        self._send(status, {"Content-Type": "application/json"},
                   json.dumps(error, separators=(",", ":")))

    def do_GET(self):
        state = self.state
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._send(200, {"Content-Type": "application/json"}, b'{"ok": true}')
            return
        if path == "/v1/models":
            models = [model for model in state["models"] if model["free"]]
            if self.headers.get("x-api-key") is not None:
                payload = models_anthropic_payload(models, state["brief"])
            else:
                payload = models_openai_payload(models)
            self._send(200, {"Content-Type": "application/json"},
                       json.dumps(payload, separators=(",", ":")))
            return
        self._error(404, "not_found", "no such route: " + path)

    def do_POST(self):
        state = self.state
        path = self.path.split("?", 1)[0]
        body, ok = self._read_json()
        if not ok or not isinstance(body, dict):
            self._error(400, "invalid_json", "the request body is not a JSON object")
            return
        if path == "/v1/chat/completions":
            status, headers, payload = serve_openai(body, state)
            self._send(status, headers, payload)
            return
        if path == "/v1/messages":
            status, headers, payload = serve_anthropic(body, state)
            self._send(status, headers, payload)
            return
        self._error(404, "not_found", "no such route: " + path)


def _http_json(method, url, payload, timeout):
    """One JSON call to the env: (status, raw bytes); None on transport failure."""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_GATEWAY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read(MAX_GATEWAY_BYTES + 1)
        except Exception:
            return exc.code, b""
    except Exception:
        return None, b""


def register_with_env(env_url, port, catalog):
    """POST the port and catalog to the env; True when it answers ok."""
    for _attempt in range(3):
        status, raw = _http_json("POST", env_url.rstrip("/") + "/register",
                                 {"protocol_version": 1, "port": port, "catalog": catalog}, 10)
        if status == 200:
            try:
                value = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                value = None
            if isinstance(value, dict) and value.get("ok") is True:
                return True
        time.sleep(WORK_POLL_SECONDS)
    return False


def work_is_done(env_url):
    """True when the env's /work answer says the trial is done."""
    status, raw = _http_json("GET", env_url.rstrip("/") + "/work", None, 10)
    if status != 200:
        return False
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return False
    return isinstance(value, dict) and value.get("done") is True


def make_state(brief):
    """The trial state shared by every request the server answers."""
    return {
        "brief": brief,
        "models": [model for model in brief["models"] if model["free"]],
        "cooled": set(),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    }


def main(argv):
    brief = read_brief(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
    text = catalog_text(brief)
    if "--list" in argv:
        sys.stdout.write(text + "\n")
        return 0
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if not env_url:
        sys.stdout.write(text + "\n")
        return 0
    state = make_state(brief)
    AggregatorHandler.state = state
    server = ThreadingHTTPServer(("0.0.0.0", 0), AggregatorHandler)
    port = server.server_address[1]
    if not register_with_env(env_url, port, json.loads(text)):
        sys.stderr.write("free-model-aggregator-02: the env refused the registration\n")
        server.server_close()
        return 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    while not work_is_done(env_url):
        time.sleep(WORK_POLL_SECONDS)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))





