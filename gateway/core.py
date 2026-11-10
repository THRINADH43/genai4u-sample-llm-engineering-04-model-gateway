"""Retries, timeouts, spending limits per caller, and logging without personal data."""

import json
import logging
import re
import threading
import time

from gateway.providers import Completion, ProviderError

log = logging.getLogger("gateway")

# USD per 1M tokens (input, output). Unknown models count as free.
PRICES = {"openai/gpt-oss-20b": (0.075, 0.30), "openai/gpt-oss-120b": (0.15, 0.60)}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
AADHAAR = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")


def redact(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = AADHAAR.sub("[id-number]", text)
    return PHONE.sub("[phone]", text)


class BudgetExceeded(Exception):
    pass


class Budget:
    """Spending limit per caller in USD. Thread-safe, because the server handles requests in
    several threads at once."""

    def __init__(self, limit_usd: float):
        self.limit = limit_usd
        self.spent: dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, caller: str):
        with self._lock:
            if self.spent.get(caller, 0.0) >= self.limit:
                raise BudgetExceeded(f"caller {caller!r} has used its ${self.limit:.2f} limit")

    def add(self, caller: str, cost: float):
        with self._lock:
            self.spent[caller] = self.spent.get(caller, 0.0) + cost


def cost_of(model: str, c: Completion) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return c.input_tokens / 1e6 * pin + c.output_tokens / 1e6 * pout


class Gateway:
    def __init__(self, provider, budget: Budget, retries=2, timeout=20.0, sleep=time.sleep):
        self.provider = provider
        self.budget = budget
        self.retries = retries
        self.timeout = timeout
        self._sleep = sleep

    def _log(self, caller, prompt, c: Completion | None, cost, started, ok, error=""):
        log.info(json.dumps({
            "caller": caller, "provider": self.provider.name, "model": self.provider.model,
            "input_tokens": c.input_tokens if c else 0, "output_tokens": c.output_tokens if c else 0,
            "cost_usd": round(cost, 6), "latency_ms": int((time.monotonic() - started) * 1000),
            "ok": ok, "error": error, "prompt_preview": redact(prompt[:80]),
        }))

    def complete(self, caller: str, prompt: str) -> tuple[Completion, float]:
        self.budget.check(caller)
        started = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                c = self.provider.complete(prompt, timeout=self.timeout)
                break
            except ProviderError as err:
                if attempt == self.retries:
                    self._log(caller, prompt, None, 0.0, started, False, str(err))
                    raise
                self._sleep(0.5 * 2 ** attempt)  # 0.5s, 1s, 2s ...
        cost = cost_of(self.provider.model, c)
        self.budget.add(caller, cost)
        self._log(caller, prompt, c, cost, started, True)
        return c, cost

    def stream(self, caller: str, prompt: str):
        """Yields text pieces. Retries only before the first piece is sent; after that a
        failure is passed on, because the caller has already shown part of the answer."""
        self.budget.check(caller)
        started = time.monotonic()
        pieces = []
        for attempt in range(self.retries + 1):
            try:
                for piece in self.provider.stream(prompt, timeout=self.timeout):
                    pieces.append(piece)
                    yield piece
                break
            except ProviderError:
                if pieces or attempt == self.retries:
                    raise
                self._sleep(0.5 * 2 ** attempt)
        text = "".join(pieces)
        c = Completion(text, len(prompt.split()), len(text.split()))  # estimate: streams give no usage
        cost = cost_of(self.provider.model, c)
        self.budget.add(caller, cost)
        self._log(caller, prompt, c, cost, started, True)
