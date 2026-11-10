"""HTTP API.

POST /chat   {"messages": [{"role": "user", "content": "..."}]}   header X-Caller: <name>
  -> 200 {"reply", "input_tokens", "output_tokens", "cost_usd"}
POST /chat   {"messages": [...], "stream": true}
  -> text/event-stream, one "data:" line per piece as it arrives, then "data: [DONE]"
429 when the caller's spending limit is used up, 502 when the model fails.
"""

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gateway.core import Budget, BudgetExceeded, Gateway
from gateway.providers import ProviderError, provider_from_config


def make_handler(gw: Gateway):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            caller = self.headers.get("X-Caller", "").strip()
            if not caller:
                return self._json(400, {"error": "X-Caller header is required"})
            if self.path != "/chat":
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                messages = body["messages"]
                assert isinstance(messages, list) and messages
                assert all(isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
                           and isinstance(m.get("content"), str) for m in messages)
            except (ValueError, KeyError, AssertionError, TypeError):
                return self._json(400, {"error": "body must be JSON with a non-empty messages list"})
            try:
                if not body.get("stream"):
                    c, cost = gw.complete(caller, messages)
                    return self._json(200, {"reply": c.text, "input_tokens": c.input_tokens,
                                            "output_tokens": c.output_tokens, "cost_usd": cost})
                gw.budget.check(caller)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for piece in gw.stream(caller, messages):
                        self.wfile.write(f"data: {json.dumps(piece)}\n\n".encode())
                        self.wfile.flush()
                except ProviderError:
                    self.wfile.write(b"data: [ERROR]\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            except BudgetExceeded as err:
                return self._json(429, {"error": str(err)})
            except ProviderError as err:
                return self._json(502, {"error": f"the model failed: {err}"})

        def log_message(self, fmt, *args):
            pass  # the gateway writes its own structured logs

    return Handler


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gw = Gateway(provider_from_config(), Budget(float(os.environ.get("GATEWAY_BUDGET_USD", "1.0"))),
                 timeout=float(os.environ.get("GATEWAY_TIMEOUT", "20")))
    port = int(os.environ.get("GATEWAY_PORT", "8090"))
    print(f"gateway on http://127.0.0.1:{port} using {gw.provider.name}:{gw.provider.model}")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(gw)).serve_forever()


if __name__ == "__main__":
    main()
