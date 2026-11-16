# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/skykc/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/skykc/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that puts OpenRouter's free models behind a single endpoint: point any
OpenAI or Anthropic client at it and every answer is a live free-model completion.

## Run it

```sh
python3 main.py            # trial mode: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
python3 main.py --list     # print the catalog document and exit 0
```

Python 3.13, standard library only — nothing is installed before the server runs.

## Routes

- `POST /v1/chat/completions` — OpenAI wire, JSON and streaming SSE
- `POST /v1/messages` — Anthropic wire, JSON and streaming SSE (translated to Chat
  Completions for the gateway and back)
- `GET /v1/models` — OpenAI list shape for `Authorization: Bearer`, Anthropic list
  shape for `x-api-key`
- `GET /healthz` — `{"ok": true}`

## How it picks a model

Every request body names `model: "auto"`. The server replaces it with the best
free model from the trial brief that meets the prompt's requirements, ranked by
the house rating (`rating_milli`, highest first; context length, then brief order
as deterministic tie-breakers). JSON-mode prompts only see JSON-capable models;
tools prompts only see tools-capable ones. The `model` field of every answer and
every streamed chunk names the model actually dialled.

## Failover

A gateway 4xx/5xx never ends a prompt on the spot: the server tries the next
best model that meets the requirements (at most 3 candidates per prompt), and if
every candidate was refused with a 429/5xx it retries the best once after a
fixed 2 s pause. A model that answered 429 is cooled down for the rest of the
trial. Refusals are reported in the `x-makefun-upstream-failures` header
(a JSON array of at most 8 `{"status", "code", "model"}` rows).

## Determinism

Every choice is a function of the brief, the requests and the gateway's answers:
no wall clock and no unseeded randomness ever reaches a request. The server
polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when it says done.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
