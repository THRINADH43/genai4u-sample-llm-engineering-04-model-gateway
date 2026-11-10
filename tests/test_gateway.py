import json
import logging
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from evaluate import load_cases, passes, run
from gateway.core import Budget, BudgetExceeded, Gateway, redact
from gateway.providers import FakeProvider, OpenAICompatibleProvider, ProviderError, provider_from_config


def no_sleep(_):
    pass


def msg(text):
    return [{"role": "user", "content": text}]


class ConfigTests(unittest.TestCase):
    def test_provider_changes_with_configuration_only(self):
        self.assertIsInstance(provider_from_config({}), FakeProvider)
        p = provider_from_config({"MODEL_PROVIDER": "groq", "MODEL_NAME": "m", "GROQ_API_KEY": "x"})
        self.assertIsInstance(p, OpenAICompatibleProvider)
        self.assertEqual((p.name, p.model), ("groq", "m"))

    def test_missing_key_and_unknown_provider(self):
        with self.assertRaises(ValueError):
            provider_from_config({"MODEL_PROVIDER": "groq"})
        with self.assertRaises(ValueError):
            provider_from_config({"MODEL_PROVIDER": "nope"})


class ReliabilityTests(unittest.TestCase):
    def test_retries_then_succeeds(self):
        fake = FakeProvider(fail_times=2)
        c, _ = Gateway(fake, Budget(1), retries=2, sleep=no_sleep).complete("a", msg("hi"))
        self.assertEqual(c.text, "echo: hi")
        self.assertEqual(fake.calls, 3)

    def test_bad_key_is_not_retried(self):
        class BadKey(FakeProvider):
            def complete(self, messages, timeout):
                self.calls += 1
                raise ProviderError("HTTP 401", retryable=False)
        fake = BadKey()
        with self.assertRaises(ProviderError):
            Gateway(fake, Budget(1), retries=2, sleep=no_sleep).complete("a", msg("hi"))
        self.assertEqual(fake.calls, 1)

    def test_gives_up_after_retries(self):
        fake = FakeProvider(fail_times=5)
        with self.assertRaises(ProviderError):
            Gateway(fake, Budget(1), retries=2, sleep=no_sleep).complete("a", msg("hi"))
        self.assertEqual(fake.calls, 3)

    def test_timeout_is_a_retryable_failure(self):
        fake = FakeProvider(delay=0.2)
        with self.assertRaises(ProviderError):
            Gateway(fake, Budget(1), retries=0, timeout=0.05).complete("a", msg("hi"))

    def test_budget_per_caller(self):
        fake = FakeProvider(model="openai/gpt-oss-20b")
        budget = Budget(0.000001)
        gw = Gateway(fake, budget, sleep=no_sleep)
        gw.complete("alice", msg("word " * 50))
        with self.assertRaises(BudgetExceeded):
            gw.complete("alice", msg("again"))
        gw.complete("bob", msg("bob still has budget"))

    def test_stream_yields_pieces(self):
        gw = Gateway(FakeProvider(answers={"hi": "hello there friend"}), Budget(1))
        self.assertEqual("".join(gw.stream("a", msg("hi"))).strip(), "hello there friend")


class LoggingTests(unittest.TestCase):
    def test_redact(self):
        text = "mail ravi.k@example.com or call +91 9876543210, aadhaar 1234 5678 9012"
        out = redact(text)
        for secret in ("ravi.k@example.com", "9876543210", "1234 5678 9012"):
            self.assertNotIn(secret, out)

    def test_logs_have_tokens_latency_cost_and_no_personal_data(self):
        with self.assertLogs("gateway", level=logging.INFO) as logs:
            Gateway(FakeProvider(), Budget(1)).complete("a", msg("my email is a@b.com"))
        record = json.loads(logs.records[0].getMessage())
        for key in ("input_tokens", "output_tokens", "latency_ms", "cost_usd"):
            self.assertIn(key, record)
        self.assertNotIn("a@b.com", logs.output[0])


class EvaluationTests(unittest.TestCase):
    def test_at_least_30_cases(self):
        self.assertGreaterEqual(len(load_cases()), 30)

    def test_scoring(self):
        self.assertTrue(passes("The answer is New Delhi.", ["delhi"]))
        self.assertFalse(passes("Mumbai", ["delhi"]))
        cases = [{"id": "1", "prompt": "capital", "expected": ["delhi"]},
                 {"id": "2", "prompt": "planet", "expected": ["mars"]}]
        result = run(Gateway(FakeProvider(answers={"capital": "Delhi"}), Budget(1)), cases)
        self.assertEqual(result["accuracy"], 0.5)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gateway.server import make_handler
        cls.gw = Gateway(FakeProvider(answers={"hi": "hello"}), Budget(1), sleep=no_sleep)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.gw))
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def post(self, path, body, caller="app1"):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"X-Caller": caller} if caller else {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()

    def test_complete(self):
        status, body = self.post("/chat", {"messages": msg("hi")})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["reply"], "hello")

    def test_stream(self):
        status, body = self.post("/chat", {"messages": msg("hi"), "stream": True})
        self.assertEqual(status, 200)
        self.assertIn("data: [DONE]", body)

    def test_bad_requests(self):
        self.assertEqual(self.post("/chat", {"messages": msg("hi")}, caller=None)[0], 400)
        self.assertEqual(self.post("/chat", {})[0], 400)
        self.assertEqual(self.post("/chat", {"messages": [{"role": "robot", "content": "x"}]})[0], 400)

    def test_spending_limit_answers_429(self):
        self.gw.budget.spent["rich-app"] = 99.0
        status, body = self.post("/chat", {"messages": msg("hi")}, caller="rich-app")
        self.assertEqual(status, 429)
        self.assertIn("limit", json.loads(body)["error"])

    def test_model_failure_answers_502(self):
        class Broken(FakeProvider):
            def complete(self, messages, timeout):
                raise ProviderError("HTTP 401", retryable=False)
        old = self.gw.provider
        self.gw.provider = Broken()
        try:
            self.assertEqual(self.post("/chat", {"messages": msg("hi")})[0], 502)
        finally:
            self.gw.provider = old


if __name__ == "__main__":
    unittest.main()
