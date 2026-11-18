"""Triage backends: a rule baseline, a local LLM through Ollama, and Claude through the Anthropic API.

Every backend takes one alert (a dict) and returns a Decision.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, describe_alert


@dataclass
class Decision:
    decision: str  # "escalate" or "close"
    risk_score: int  # 0-100
    reason: str


def _parse_decision(text: str) -> Decision:
    data = json.loads(text)
    decision = str(data["decision"]).lower()
    if decision not in {"escalate", "close"}:
        raise ValueError(f"unexpected decision: {decision!r}")
    score = max(0, min(100, int(data["risk_score"])))
    return Decision(decision, score, str(data["reason"]).strip())


class BaselineBackend:
    """What a simple scoring rule does: count fired rules and escalate at a threshold.

    This stands in for classic rule-based monitoring and is the bar any LLM has to beat.
    """

    name = "baseline"

    def __init__(self, threshold: int = 2):
        self.threshold = threshold

    def triage(self, alert: dict) -> Decision:
        rules = alert["rules_fired"].split(",")
        score = min(100, 25 * len(rules))
        decision = "escalate" if len(rules) >= self.threshold else "close"
        return Decision(decision, score, f"{len(rules)} rule(s) fired: {', '.join(rules)}")


class OllamaBackend:
    """A free LLM running locally through Ollama (https://ollama.com). No API key needed."""

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"

    def triage(self, alert: dict) -> Decision:
        body = {
            "model": self.model,
            "stream": False,
            "format": OUTPUT_SCHEMA,  # Ollama constrains the reply to this JSON schema
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": describe_alert(alert)},
            ],
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it installed and running, "
                f"and have you pulled the model (ollama pull {self.model})?"
            ) from exc
        return _parse_decision(payload["message"]["content"])


class ClaudeBackend:
    """Claude through the Anthropic API. Needs ANTHROPIC_API_KEY (or an `ant auth login` profile)."""

    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic  # imported here so the other backends work without the package

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"claude:{model}"

    def triage(self, alert: dict) -> Decision:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": describe_alert(alert)}],
            output_config={
                "effort": "low",  # a short classification task; raise it only if measured accuracy improves
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined alert {alert['alert_id']}")
        text = next(block.text for block in response.content if block.type == "text")
        return _parse_decision(text)


def make_backend(spec: str):
    """Build a backend from 'baseline', 'ollama[:model]' or 'claude[:model]'."""
    kind, _, model = spec.partition(":")
    if kind == "baseline":
        return BaselineBackend()
    if kind == "ollama":
        return OllamaBackend(model) if model else OllamaBackend()
    if kind == "claude":
        return ClaudeBackend(model) if model else ClaudeBackend()
    raise ValueError(f"unknown backend {spec!r}; use baseline, ollama[:model] or claude[:model]")
