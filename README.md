# Evaluated model gateway service

An HTTP service that sits between applications and language models. Apps send a prompt; the
gateway picks the model from configuration, retries failures, enforces a spending limit per
caller, logs tokens, latency and cost without personal data, and can stream the answer.
An evaluation command scores any configured model on 32 test cases and compares it with the
last run. Standard library only.

## Run

```
python -m gateway.server                                   # fake model, no key needed
GATEWAY_PROVIDER=groq GROQ_API_KEY=... python -m gateway.server
```

```
curl -X POST localhost:8090/v1/complete -H "X-Caller: shop-app" -d '{"prompt": "Say hi"}'
curl -N -X POST localhost:8090/v1/stream -H "X-Caller: shop-app" -d '{"prompt": "Say hi"}'
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `GATEWAY_PROVIDER` | `fake` | `fake`, `groq` or `openrouter` |
| `GATEWAY_MODEL` | `openai/gpt-oss-20b` | any model id of that provider |
| `GATEWAY_BUDGET_USD` | `1.0` | spending limit per caller |
| `GATEWAY_TIMEOUT` | `20` | seconds per model call |
| `GATEWAY_PORT` | `8090` | |

Keys are read from `GROQ_API_KEY` / `OPENROUTER_API_KEY` and never logged or committed.

## Design

- `gateway/providers.py` is the model layer. `FakeProvider` and `OpenAICompatibleProvider`
  have the same `complete` and `stream` methods. `provider_from_config` builds one from
  environment variables, so changing model or provider is a configuration change only.
- `gateway/core.py` has the `Gateway` (retries with backoff, timeout, budget checks, logging),
  the thread-safe `Budget`, prices, and `redact`.
- `gateway/server.py` is the HTTP API (`ThreadingHTTPServer`).

## Reliability

- **Timeouts:** every model call has a timeout; a timeout counts as a retryable failure.
- **Retries:** timeouts, HTTP 429 and 5xx are retried twice with backoff (0.5 s, 1 s). Other
  errors, like a bad key, fail at once. A stream is retried only before its first piece; after
  that the client gets `data: [ERROR]`, because part of the answer was already shown.
- **Spending limit:** each `X-Caller` has its own budget. Once used up, the API answers 402.
- Status codes: 200, 400 (no caller or no prompt), 402 (budget), 404, 502 (model kept failing).

## Logs

One JSON line per call: caller, provider, model, input/output tokens, cost, latency, ok/error
and an 80-character prompt preview with emails, Indian phone numbers and Aadhaar-style
12-digit numbers replaced by `[email]`, `[phone]`, `[id-number]`. Streams do not report
token usage, so their token counts are word-count estimates.

## Evaluation

`data/eval.jsonl` has 32 cases (general knowledge, arithmetic, programming, Indian
geography). A case passes if all expected words appear in the answer.

```
GATEWAY_PROVIDER=groq python evaluate.py
python evaluate.py --fail-under 0.9      # exits 1 below 0.9
```

Results are saved to `results/<provider>_<model>.json`; the next run prints the change and
the cases that started failing. Two real runs of `openai/gpt-oss-20b` on Groq:

```
groq:openai/gpt-oss-20b accuracy 0.938 on 32 cases, cost $0.00046, 8.9s
previous accuracy 0.938 -> now 0.969 (+0.031)
groq:openai/gpt-oss-20b accuracy 0.969 on 32 cases, cost $0.00043, 9.2s
```

The score moved between runs with no change in code, so differences of one case (3%) are
noise; I would only trust a change of several cases.

## Tests

`python -m unittest discover -s tests -t .` runs 14 tests with the fake model and no key:
configuration, retries, giving up, timeouts, per-caller budgets, streaming, redaction,
log fields, evaluation scoring and the HTTP endpoints. GitHub Actions runs them on every
push (`.github/workflows/tests.yml`).
