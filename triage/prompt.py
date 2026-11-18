"""Prompt and output schema shared by every LLM backend."""

from __future__ import annotations

from .data import HOME_COUNTRY, RULES

SYSTEM_PROMPT = f"""You are a transaction-monitoring analyst at a German bank. A rule-based system
has raised an alert on a card transaction. Most alerts from these rules are false positives,
so your job is to decide whether this alert should be ESCALATED to an investigator or CLOSED.

Weigh the evidence as a whole instead of reacting to any single rule. Typical fraud combines
several signals, for example a new device together with a large foreign transaction, or a burst
of gift-card purchases. Typical legitimate activity includes established customers spending
abroad on travel, restaurants or fuel from a known device.

Customers' home country is {HOME_COUNTRY}.

Respond with a decision ("escalate" or "close"), a risk score from 0 (certainly legitimate) to
100 (certainly fraud), and one or two sentences explaining the decision in terms an investigator
can check quickly."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["escalate", "close"]},
        "risk_score": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "risk_score", "reason"],
    "additionalProperties": False,
}


def describe_alert(alert: dict) -> str:
    """Render one alert as plain text for the model. The ground-truth label is never included."""
    ratio = alert["amount_eur"] / alert["customer_avg_amount_eur"]
    rules = "\n".join(f"- {r}: {RULES[r]}" for r in alert["rules_fired"].split(","))
    return f"""Alert {alert['alert_id']}
Rules fired:
{rules}

Transaction:
- Amount: EUR {alert['amount_eur']:.2f} ({ratio:.1f}x the customer's 90-day average of EUR {alert['customer_avg_amount_eur']:.2f})
- Country: {alert['country']}
- Merchant category: {alert['merchant_category']}
- Local time: {alert['hour']:02d}:00
- Transactions by this card in the last hour: {alert['txn_count_last_hour']}
- New device: {'yes' if alert['new_device'] else 'no'}
- Account age: {alert['account_age_days']} days"""
