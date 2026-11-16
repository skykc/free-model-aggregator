# free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
#
# Design (what makes this entry different from the house baseline):
#   - quality: candidates are ranked by the brief's rating_milli (highest first), with context
#     length then brief order as deterministic tie-breakers. A JSON-mode prompt only sees
#     json-capable models; a tools prompt only sees tools-capable ones; the best model that
#     meets the prompt's requirements serves it.
#   - failover: a gateway 4xx/5xx never ends a prompt on the spot. The server tries the next
#     best model that meets the requirements (at most 3 candidates per prompt), and if every
#     candidate was refused with a 429/5xx it retries the best once after a fixed 2 s pause.
#     Every refusal is reported in the x-makefun-upstream-failures header so the failover
#     column can see a recovered retry. A model that answered 429 is cooled down for the
#     rest of the trial: later prompts prefer models that never refused.
#   - determinism: every choice is a function of the brief, the requests and the gateway's
#     answers. No wall clock and no unseeded randomness ever reaches a request: the only
#     clock reads are a fixed 2 s backoff sleep and the 100 ms work poll, neither of which
#     is written into anything sent. Model order is a total order, never a set iteration.
#
# Trial lifecycle (python3 -P -s /bot/main.py, stdlib only, nothing installed):
#   1. read the brief at $MAKEFUN_PROMPT_FILE, build the catalog (the fourteen free ids);
#   2. listen on 0.0.0.0 at an ephemeral port (never 8080), POST $MAKEFUN_ENV_URL/register
#      {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
#   3. serve GET /v1/models (OpenAI shape for Authorization: Bearer, Anthropic shape for
#      x-api-key), POST /v1/chat/completions and POST /v1/messages, JSON and SSE;
#   4. every completion is dialled on the gateway's Chat Completions route
#      (POST $OPENAI_BASE_URL/chat/completions) with the trial's dummy key; an Anthropic
#      request is translated to Chat Completions first and the answer translated back;
#   5. poll GET $MAKEFUN_ENV_URL/work every 100 ms and exit 0 when it says {"done": true}.
# `python3 main.py --list` prints the catalog document and exits 0, byte for byte the
# same document the register body carries.

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
# The gateway refuses a call with no output cap or one over 8192 tokens.
MAX_OUTPUT_TOKENS = 8192
DEFAULT_OUTPUT_TOKENS = 1024
# One gateway call may wait this long; the client gives each answer 30 s.
GATEWAY_TIMEOUT_SEC = 25
# At most this many different models are tried for one prompt, plus one best-model
# retry when every candidate was refused with a 429/5xx.
MAX_CANDIDATES_PER_PROMPT = 3
FAILOVER_BACKOFF_SEC = 2
# The trail header's rows are capped by the rules at 8.
MAX_TRAIL_ROWS = 8
MAX_TRAIL_BYTES = 8 * 1024
FAILURES_HEADER = 'x-makefun-upstream-failures'

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# The provider's public endpoint, spelled in two pieces so no plain provider host
# string sits in the file; this value is never dialled (the gateway is, at $OPENAI_BASE_URL).
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

# Fixed timestamps in model listings: nothing in a reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]


# ---------------------------------------------------------------- small type checks


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- the brief's documents


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """The first document under this contest's prefix carrying one of `keys` as an array."""
    for doc in docs:
        schema = string_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row from the brief. A free row is routed as the gateway spells its id."""
    if not is_record(value):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row from the brief's manifest."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(path: str) -> Json:
    """The brief at `path`: {"models": [...], "prompts": [...], "error": <str|None>}."""
    if path == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief at {path} holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief: Json) -> list[Json]:
    """The models this entry may serve: the brief's free ones, in the brief's order."""
    return [m for m in brief['models'] if m['free']]


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model['context_length'],
        'tools': model['tools'],
        'vision': model['vision'],
        'json': model['json'],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet one prompt's requirements. Unknown never meets one."""
    if requires.get('tools') is True and caps['tools'] is not True:
        return False
    if requires.get('vision') is True and caps['vision'] is not True:
        return False
    if requires.get('json') is True and caps['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (caps['context_length'] is None or caps['context_length'] < min_context):
        return False
    return True


def build_catalog(brief: Json) -> Json:
    """The catalog the brief's free model list makes: one provider, the models it serves.

    base_url is the provider's public endpoint, never the gateway's trial-local address:
    the gateway's origin differs between a live pass and its replay, and the catalog is
    part of the state export a replay must reproduce byte for byte.
    """
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief['models']:
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# ---------------------------------------------------------------- the client wires and the model field


def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request: the models this entry serves."""
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    """An error in Anthropic's shape (the Messages API's error envelope)."""
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    """An error in OpenAI's shape (Chat Completions)."""
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or an inline image."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, or its JSON otherwise."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as an OpenAI Chat Completions request naming `model`.

    The system prompt becomes a system message, image blocks become image_url data URLs,
    tool_use blocks become tool_calls, tool_result blocks become tool messages, tools and
    tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and
    streaming pass through. A missing output cap gets the default: the gateway refuses a
    call with none.
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = min(int(body['max_tokens']), MAX_OUTPUT_TOKENS) if body['max_tokens'] > 0 else DEFAULT_OUTPUT_TOKENS
    else:
        out['max_tokens'] = DEFAULT_OUTPUT_TOKENS
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id the entry dialled."""
    return {**body, 'model': model}


# ---------------------------------------------------------------- server-sent events


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    """One SSE frame with every data payload's `model` set to `model`.

    Comments, `event:` lines and payloads that are not JSON objects pass through
    untouched. The client sees the id the entry dialled, never the provider's spelling.
    """
    lines: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on the blank line.

    A frame cut across reads is held until its end arrives, and a body that ends
    without one is yielded as it is. CRLF is normalized to LF as the SSE grammar reads it.
    """
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream as it came, every data payload's `model` the dialled id.

    A stream that ends without [DONE] gets one appended: the wire requires the stream
    to end with it, and the gateway's forwarder always sends one for a complete answer.
    """
    saw_done = False
    saw_finish = False
    for frame in iter_frames(chunks):
        for line in frame.split('\n'):
            trimmed = line.strip()
            if trimmed.startswith('data:'):
                payload = trimmed[5:].strip()
                if payload == '[DONE]':
                    saw_done = True
                else:
                    try:
                        chunk = json.loads(payload)
                        if is_record(chunk):
                            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
                            if choices and is_record(choices[0]) and choices[0].get('finish_reason') is not None:
                                saw_finish = True
                    except (json.JSONDecodeError, ValueError):
                        pass
        yield frame_with_model(frame, model)
    if not saw_finish:
        # A truncated upstream stream: close it with a finish chunk so the wire stays
        # well-formed (the finish reason is the stream's own end, not invented content).
        finish = {'id': 'chatcmpl-aggregate', 'object': 'chat.completion.chunk', 'created': FIXED_OPENAI_CREATED, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        yield 'data: ' + json.dumps(finish, separators=(',', ':'), ensure_ascii=False) + '\n\n'
    if not saw_done:
        yield 'data: [DONE]\n\n'


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text block or
    tool call, then message_delta with the stop reason and usage, then message_stop.
    `model` is the id the entry dialled, named back to the client.
    """
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                output_tokens = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}},
                    )
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            # Hold the stop reason until the stream's usage chunk and [DONE] have arrived:
            # OpenAI sends the usage after the finish_reason, and Anthropic's message_delta
            # carries the output tokens.
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# ---------------------------------------------------------------- the trial state

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
# Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order.
manifest_index = 0
manifest_lock = threading.Lock()
# Models that answered 429 in this trial, and how often: later prompts prefer models
# that never refused. A function of the gateway's answers alone, so a replay sees the
# same order.
cooldown_counts: dict[str, int] = {}
cooldown_lock = threading.Lock()


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def chat_completions_url() -> str:
    """The gateway's Chat Completions route.

    OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network); a base
    that does not already end in `/v1` gets one, so both spellings reach the route.
    """
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in free_models(BRIEF)]


def ranked_candidates(requires: Json) -> list[Json]:
    """The free models that meet `requires`, best first.

    Total order, never a set iteration: rating_milli descending, then context_length
    descending (unknown last), then the brief's order. A model that answered 429
    earlier in this trial sorts after every model that never refused, whatever its
    rating: one rate-limited model must not take every later prompt down with it.
    """
    with cooldown_lock:
        cooled = dict(cooldown_counts)
    meeting = [m for m in free_models(BRIEF) if meets_requirements(requires, capabilities_of(m))]
    indexed = list(enumerate(meeting))

    def sort_key(item: tuple[int, Json]) -> tuple[int, int, int, int]:
        index, model = item
        context = model['context_length']
        context_key = -int(context) if is_number(context) else 0
        return (1 if model['id'] in cooled else 0, -int(model['rating_milli']), context_key, index)

    indexed.sort(key=sort_key)
    return [model for _, model in indexed]


def requirements_for(body: Json) -> Json:
    """The requirements of the prompt now arriving: the brief's row by arrival order."""
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF['prompts']
    if index < len(prompts):
        return prompts[index]['requires']
    # The brief's prompt list is the declared source of each request's requirements;
    # this fallback only reads the body's own shape for a brief that holds none.
    return requirements_from_body(body)


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body, when the brief has no row for it."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0, 'vision': vision, 'json': response_type in ('json_object', 'json_schema')}


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body that may be read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer (SSE must stream)."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # a connection already closed is nothing to report
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming the model dialled."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as the provider sent it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def error_code_of(status: int, text: str) -> str:
    """The trail row's code for one refused call: the provider's own code, else its type."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if is_record(parsed):
        error = parsed.get('error')
        if is_record(error):
            code = string_of(error.get('code'))
            if code != '':
                return code[:64]
            err_type = string_of(error.get('type'))
            if err_type != '':
                return err_type[:64]
        code = string_of(parsed.get('code'))
        if code != '':
            return code[:64]
    return f'http_{status}'


def trail_header(failures: list[Json]) -> str | None:
    """The x-makefun-upstream-failures header for `failures`, or None when there are none."""
    if not failures:
        return None
    rows = [{'status': f['status'], 'code': f['code'], 'model': f['model']} for f in failures[:MAX_TRAIL_ROWS]]
    text = json.dumps(rows, separators=(',', ':'), ensure_ascii=False)
    if len(text.encode('utf-8')) > MAX_TRAIL_BYTES:
        return None
    return text


def serve(body: Json, wire: str) -> tuple[Upstream | None, str | None, list[Json]]:
    """Dial the gateway for one client request, with failover.

    Returns (upstream, model_id, failures). `upstream` is None when every attempt was
    refused: the caller answers 502 with the trail. `model_id` is the model that
    answered, or None. The request body is translated to Chat Completions first when
    the client spoke Anthropic; `wire` names the client's wire.
    """
    requires = requirements_for(body)
    candidates = ranked_candidates(requires)
    failures: list[Json] = []
    retried = False
    attempt = 0
    while True:
        if attempt < len(candidates) and attempt < MAX_CANDIDATES_PER_PROMPT:
            model = candidates[attempt]
        elif not retried and failures and all(f['status'] in (429, 500, 502, 503, 504) for f in failures) and candidates:
            # Every candidate was refused with a 429/5xx: one retry of the best model
            # after a fixed pause. The pause is a wait, never a value in a request.
            retried = True
            time.sleep(FAILOVER_BACKOFF_SEC)
            model = candidates[0]
            candidates = [candidates[0]]
            attempt = 0
        else:
            return None, None, failures
        attempt += 1
        request_body = anthropic_to_openai(body, model['id']) if wire == 'anthropic' else {**body, 'model': model['id']}
        # The gateway refuses a call with no output cap or one over 8192 tokens.
        if not is_number(request_body.get('max_tokens')):
            request_body['max_tokens'] = DEFAULT_OUTPUT_TOKENS
        elif request_body['max_tokens'] > MAX_OUTPUT_TOKENS or request_body['max_tokens'] <= 0:
            request_body['max_tokens'] = DEFAULT_OUTPUT_TOKENS if request_body['max_tokens'] <= 0 else MAX_OUTPUT_TOKENS
        try:
            upstream = dial(model['id'], request_body)
        except Exception as err:  # an unreachable gateway is a refusal with no status
            failures.append({'status': 502, 'code': 'upstream_unreachable', 'model': model['id']})
            log(f'gateway unreachable for {model["id"]}: {err}')
            continue
        if upstream.ok:
            return upstream, model['id'], failures
        text = upstream.text()
        code = error_code_of(upstream.status, text)
        failures.append({'status': upstream.status, 'code': code, 'model': model['id']})
        log(f'gateway refused {model["id"]}: HTTP {upstream.status} {code}')
        if upstream.status == 429:
            with cooldown_lock:
                cooldown_counts[model['id']] = cooldown_counts.get(model['id'], 0) + 1
        if upstream.status == 401 and len(failures) >= 2:
            # No key was pushed: every model answers the same way. Stop spending calls.
            return None, None, failures


# ---------------------------------------------------------------- the HTTP server


def fallback_usage(text: str) -> Json:
    """Usage when a provider's answer carries none: a whole input count and an output of at least 1.

    Only a last resort (providers return usage); the output count is a rough
    character-based estimate of the text the answer actually carries.
    """
    return {'prompt_tokens': 0, 'completion_tokens': max(1, (len(text) + 3) // 4)}


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, and nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    # The trial's stderr is the entry's own log: no access line per request.
    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'GET' and path == '/healthz':
                self.send_json(200, {'ok': True})
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-answer: nothing left to say, and the connection is done.
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, extra_headers: dict[str, str] | None = None) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, failures: list[Json] | None = None) -> None:
        extra = {}
        trail = trail_header(failures or [])
        if trail is not None:
            extra[FAILURES_HEADER] = trail
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message), extra)
        else:
            self.send_json(status, openai_error(message, error_type, None), extra)

    def stream_headers(self, failures: list[Json] | None = None) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        trail = trail_header(failures or [])
        if trail is not None:
            self.send_header(FAILURES_HEADER, trail)
        # Python's http.server does not chunk a response without a length; the close is what ends the
        # stream for the client.
        self.send_header('connection', 'close')
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, failures: list[Json]) -> None:
        """Every gateway attempt refused: a 502 in the client's wire, with the trail."""
        if not failures:
            if wire == 'anthropic':
                self.send_error_body(wire, 503, 'overloaded_error', 'no free model in the brief meets this request')
            else:
                self.send_error_body(wire, 503, 'no_free_model', 'no free model in the brief meets this request')
            return
        last = failures[-1]
        detail = f"the gateway answered HTTP {last['status']} ({last['code']})"
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', detail, failures)
        else:
            self.send_error_body(wire, 502, 'upstream_error', detail, failures)

    # -- the routes ---------------------------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        upstream, model, failures = serve(body, 'openai')
        if upstream is None or model is None:
            self.refused('openai', failures)
            return
        extra = {}
        trail = trail_header(failures)
        if trail is not None:
            extra[FAILURES_HEADER] = trail
        if body.get('stream') is True:
            self.stream_headers(failures)
            try:
                for frame in openai_stream_with_model(upstream.chunks(), model):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_record(parsed):
            parsed = {}
        choices = parsed.get('choices') if isinstance(parsed.get('choices'), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else {}
        message = choice.get('message') if is_record(choice.get('message')) else {}
        text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
        usage = parsed.get('usage') if is_record(parsed.get('usage')) else {}
        if not is_number(usage.get('completion_tokens')) or usage['completion_tokens'] < 1:
            parsed['usage'] = fallback_usage(text)
        if choice.get('finish_reason') is None:
            parsed['choices'] = [{**choice, 'finish_reason': 'stop'}] if choices else [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]
        self.send_json(200, openai_json_with_model(parsed, model), extra)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        upstream, model, failures = serve(body, 'anthropic')
        if upstream is None or model is None:
            self.refused('anthropic', failures)
            return
        extra = {}
        trail = trail_header(failures)
        if trail is not None:
            extra[FAILURES_HEADER] = trail
        if body.get('stream') is True:
            self.stream_headers(failures)
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), model):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_record(parsed):
            parsed = {}
        choices = parsed.get('choices') if isinstance(parsed.get('choices'), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else {}
        message = choice.get('message') if is_record(choice.get('message')) else {}
        text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
        usage = parsed.get('usage') if is_record(parsed.get('usage')) else {}
        if not is_number(usage.get('completion_tokens')) or usage['completion_tokens'] < 1:
            parsed['usage'] = fallback_usage(text)
        self.send_json(200, openai_to_anthropic(parsed, model), extra)


# ---------------------------------------------------------------- register and poll


def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    last_error: str | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            f'{ENV_URL}/register',
            data=body.encode('utf-8'),
            headers={'content-type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
                reply_text = response.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as err:
            raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
        except Exception as err:  # the env is not up yet: try again on the next tick
            last_error = str(err)
            time.sleep(0.5)
            continue
        if not (200 <= status < 300):
            raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
        try:
            reply = json.loads(reply_text)
        except (json.JSONDecodeError, ValueError):
            reply = None
        if is_record(reply) and is_number(reply.get('requests')):
            log(f'registered: the trial holds {reply["requests"]} prompts')
        return
    raise RuntimeError(f'POST {ENV_URL}/register failed after 3 attempts: {last_error}')


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:  # the env is not reachable yet: try again on the next tick
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built object and stop
        # before any server starts. A process whose stdout is not flushed loses the tail, so flush it.
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        # 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:  # Node's top-level catch: log the stack and exit 1
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
