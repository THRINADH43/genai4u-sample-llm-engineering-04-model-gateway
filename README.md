# Evaluated model gateway service

An HTTP service that sits between applications and language models. Apps send a prompt; the
gateway picks the model from configuration, retries failures, enforces a spending limit per
caller, logs tokens, latency and cost without personal data, and can stream the answer.
An evaluation command scores any configured model on 32 test cases and compares it with the
last run. Standard library only.

## Run

```
python -m gateway.server                                   # fake model, no key needed
MODEL_PROVIDER=groq GROQ_API_KEY=... python -m gateway.server
```

```
curl -X POST localhost:8090/chat -H "X-Caller: shop-app" \
     -d '{"messages": [{"role": "user", "content": "Say hi"}]}'
# 200 {"reply": "...", "input_tokens": ..., "output_tokens": ..., "cost_usd": ...}
curl -N -X POST localhost:8090/chat -H "X-Caller: shop-app" \
     -d '{"messages": [{"role": "user", "content": "Say hi"}], "stream": true}'
# data: "Hi" ... data: [DONE]
```

Settings (see `.env.example`; copy it to `.env` or export the variables, and never commit the key):

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_PROVIDER` | `fake` | `fake`, `groq` or `openrouter` |
| `MODEL_NAME` | `openai/gpt-oss-20b` | any model id of that provider |
| `GATEWAY_BUDGET_USD` | `1.0` | spending limit per caller |
| `GATEWAY_TIMEOUT` | `20` | seconds per model call |
| `GATEWAY_PORT` | `8090` | |

Keys are read from `GROQ_API_KEY` / `OPENROUTER_API_KEY` and never logged or committed.
Standard library only, so `requirements.txt` is empty.

## Design

- `gateway/providers.py` is the model layer. `FakeProvider` and `OpenAICompatibleProvider`
  have the same `complete` and `stream` methods. `provider_from_config` builds one from
  environment variables, so changing model or provider is a configuration change only.
- `gateway/core.py` has the `Gateway` (retries with backoff, timeout, budget checks, logging),
  the thread-safe `Budget`, prices, and `redact`.
- `gateway/server.py` is the HTTP API (`ThreadingHTTPServer`): one endpoint, `POST /chat`.

## Reliability

- **Timeouts:** every model call has a timeout; a timeout counts as a retryable failure.
- **Retries:** timeouts, HTTP 429 and 5xx are retried twice with backoff (0.5 s, 1 s). Other
  errors, like a bad key (401), are marked not retryable and fail at once with a 502. A stream is retried only before its first piece; after
  that the client gets `data: [ERROR]`, because part of the answer was already shown.
- **Spending limit:** each `X-Caller` has its own budget. Once used up, the API answers 429
  with a message saying the caller's limit is used.
- Status codes: 200, 400 (no caller, or no valid `messages`), 404, 429 (spending limit), 502
  (the model failed).

## Logs

One JSON line per call: caller, provider, model, input/output tokens, cost, latency, ok/error
and an 80-character preview of the last user message, with emails, Indian phone numbers (also written `98765 43210`) and Aadhaar-style
12-digit numbers replaced by `[email]`, `[phone]`, `[id-number]`. Masking happens before the
cut, so a number at the 80-character edge cannot leak half-masked. Failed calls are logged too. Streams do not report
token usage, so their token counts are word-count estimates.

## Evaluation

`data/eval.json` has 32 cases (general knowledge, arithmetic, programming, Indian
geography). A case passes if all expected words appear in the answer.

```
MODEL_PROVIDER=groq python evaluate.py
python evaluate.py --fail-under 0.9      # exits 1 below 0.9
```

Results are saved to `results/<provider>_<model>.json`; the next run prints the change and
the cases that started failing. Two real runs of `openai/gpt-oss-20b` on Groq:

```
groq:openai/gpt-oss-20b accuracy 0.938 on 32 cases, cost $0.00046, 8.9s
previous accuracy 0.938 -> now 0.969 (+0.031)
groq:openai/gpt-oss-20b accuracy 0.969 on 32 cases, cost $0.00043, 9.2s
```

A third run, after moving to the `/chat` message format:

```
previous accuracy 0.969 -> now 0.938 (-0.031)
now failing: q15
groq:openai/gpt-oss-20b accuracy 0.938 on 32 cases, cost $0.00043, 6.8s
```

The score moved between runs with no change in code, so differences of one case (3%) are
noise; I would only trust a change of several cases.

## Tests

`python -m unittest discover -s tests -t .` runs 17 tests with the fake model and no key:
configuration, retries, no retry for a bad key, giving up, timeouts, per-caller budgets, streaming, redaction,
log fields, evaluation scoring and the HTTP endpoint (200, stream, 400, 429 and 502). GitHub Actions runs them on every
push (`.github/workflows/tests.yml`).
