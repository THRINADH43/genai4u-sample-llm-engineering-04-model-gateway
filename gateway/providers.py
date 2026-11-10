"""The model layer. Every provider has the same two methods, so the rest of the code never
knows which one it is talking to. Which provider and model to use comes from configuration."""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderError(Exception):
    """Raised for failures worth retrying (timeouts, 429, 5xx)."""


class Provider(Protocol):
    name: str
    model: str

    def complete(self, prompt: str, timeout: float) -> Completion: ...
    def stream(self, prompt: str, timeout: float) -> Iterator[str]: ...


class FakeProvider:
    """Answers from a small table, or echoes. Used in tests and when no key is set.
    fail_times makes the first N calls fail, to test retries."""

    name = "fake"

    def __init__(self, model="fake-1", answers=None, fail_times=0, delay=0.0):
        self.model = model
        self.answers = answers or {}
        self.fail_times = fail_times
        self.delay = delay
        self.calls = 0

    def _answer(self, prompt):
        for key, value in self.answers.items():
            if key.lower() in prompt.lower():
                return value
        return f"echo: {prompt}"

    def complete(self, prompt, timeout):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("fake failure")
        if self.delay > timeout:
            time.sleep(timeout)
            raise ProviderError("timed out")
        time.sleep(self.delay)
        text = self._answer(prompt)
        return Completion(text, len(prompt.split()), len(text.split()))

    def stream(self, prompt, timeout):
        for word in self.complete(prompt, timeout).text.split(" "):
            yield word + " "


class OpenAICompatibleProvider:
    """Any OpenAI-style chat API: Groq, OpenRouter, a local server. Uses only urllib."""

    def __init__(self, name, base_url, model, api_key):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._key = api_key

    def _request(self, prompt, stream):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": stream,
                "reasoning_effort": "low"}
        return urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                     "User-Agent": "model-gateway/1.0"})

    def complete(self, prompt, timeout):
        try:
            with urllib.request.urlopen(self._request(prompt, False), timeout=timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as err:
            if err.code == 429 or err.code >= 500:
                raise ProviderError(f"HTTP {err.code}") from err
            raise
        except (urllib.error.URLError, TimeoutError) as err:
            raise ProviderError(str(err)) from err
        usage = data.get("usage", {})
        return Completion(data["choices"][0]["message"]["content"],
                          usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    def stream(self, prompt, timeout):
        try:
            r = urllib.request.urlopen(self._request(prompt, True), timeout=timeout)
        except (urllib.error.URLError, TimeoutError) as err:
            raise ProviderError(str(err)) from err
        with r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def provider_from_config(env=os.environ) -> Provider:
    """GATEWAY_PROVIDER=fake|groq|openrouter and GATEWAY_MODEL pick the model. Changing the
    model is a configuration change only."""
    kind = env.get("GATEWAY_PROVIDER", "fake")
    model = env.get("GATEWAY_MODEL", "fake-1" if kind == "fake" else "openai/gpt-oss-20b")
    if kind == "fake":
        return FakeProvider(model, answers=json.loads(env.get("GATEWAY_FAKE_ANSWERS", "{}")))
    if kind not in PROVIDERS:
        raise ValueError(f"unknown provider {kind!r}; use one of fake, {', '.join(PROVIDERS)}")
    base_url, key_var = PROVIDERS[kind]
    key = env.get(key_var)
    if not key:
        raise ValueError(f"{key_var} is not set")
    return OpenAICompatibleProvider(kind, base_url, model, key)
