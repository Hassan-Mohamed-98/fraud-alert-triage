"""Synthetic transaction-monitoring alerts with a hidden ground truth.

Rule-based monitoring systems fire on simple conditions (a large amount, a
foreign transaction, a burst of activity) and most of the alerts they raise
turn out to be false positives. This module imitates that: it generates
customer transactions, fires alerts with simple rules, and decides whether
each alerted transaction is really fraud from a noisy combination of risk
factors that no single rule sees.

All data is synthetic. No real customer or bank data is used.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOME_COUNTRY = "DE"
FOREIGN_COUNTRIES = ["AT", "FR", "NL", "ES", "IT", "PL", "TR", "AE", "NG", "BR", "US", "CN"]
HIGH_RISK_COUNTRIES = {"NG", "BR", "AE"}

MERCHANT_CATEGORIES = {
    # category: (relative frequency, inherent risk weight)
    "groceries": (0.22, 0.0),
    "restaurants": (0.14, 0.0),
    "travel": (0.10, 0.3),
    "online_retail": (0.16, 0.4),
    "electronics": (0.10, 0.9),
    "fuel": (0.08, 0.0),
    "gift_cards": (0.06, 1.6),
    "crypto_exchange": (0.05, 1.4),
    "money_transfer": (0.09, 1.1),
}

RULES = {
    "HIGH_AMOUNT": "Amount is more than 4x the customer's 90-day average",
    "FOREIGN_TXN": "Card used outside the customer's home country",
    "VELOCITY": "More than 4 transactions in the last hour",
    "NEW_DEVICE": "Transaction from a device not seen before for this customer",
    "RISKY_MERCHANT": "Merchant category on the high-risk watch list",
}


def _fired_rules(row: dict) -> list[str]:
    rules = []
    if row["amount_eur"] > 4 * row["customer_avg_amount_eur"]:
        rules.append("HIGH_AMOUNT")
    if row["country"] != HOME_COUNTRY:
        rules.append("FOREIGN_TXN")
    if row["txn_count_last_hour"] > 4:
        rules.append("VELOCITY")
    if row["new_device"]:
        rules.append("NEW_DEVICE")
    if row["merchant_category"] in {"gift_cards", "crypto_exchange", "money_transfer"}:
        rules.append("RISKY_MERCHANT")
    return rules


def _fraud_probability(row: dict) -> float:
    """Hidden ground truth: fraud is driven by combinations of signals, not single rules."""
    ratio = row["amount_eur"] / row["customer_avg_amount_eur"]
    foreign = row["country"] != HOME_COUNTRY
    night = row["hour"] < 6
    logit = -4.2
    logit += 0.9 * math.log1p(max(ratio - 1, 0))
    logit += 0.6 if foreign else 0.0
    logit += 1.0 if row["country"] in HIGH_RISK_COUNTRIES else 0.0
    logit += 1.1 if row["new_device"] else 0.0
    logit += 0.7 if night else 0.0
    logit += 0.25 * max(row["txn_count_last_hour"] - 2, 0)
    logit += MERCHANT_CATEGORIES[row["merchant_category"]][1]
    logit += 0.9 if row["account_age_days"] < 60 else 0.0
    # Combinations that analysts recognise as typical fraud patterns
    logit += 1.2 if (row["new_device"] and foreign and ratio > 3) else 0.0
    logit += 1.0 if (row["merchant_category"] == "gift_cards" and row["txn_count_last_hour"] > 4) else 0.0
    # Typical legitimate pattern: an established customer travelling
    logit -= 1.3 if (foreign and not row["new_device"] and row["merchant_category"] in {"travel", "restaurants", "fuel"}) else 0.0
    return 1.0 / (1.0 + math.exp(-logit))


def _transaction(rng: np.random.Generator, idx: int) -> dict:
    categories = list(MERCHANT_CATEGORIES)
    weights = np.array([MERCHANT_CATEGORIES[c][0] for c in categories])
    avg = float(np.round(rng.lognormal(mean=4.0, sigma=0.6), 2))
    spike = rng.random() < 0.35
    amount = avg * (rng.uniform(4, 20) if spike else rng.lognormal(0, 0.5))
    foreign = rng.random() < 0.3
    return {
        "alert_id": f"ALRT-{idx:05d}",
        "amount_eur": float(np.round(amount, 2)),
        "customer_avg_amount_eur": avg,
        "country": str(rng.choice(FOREIGN_COUNTRIES)) if foreign else HOME_COUNTRY,
        "merchant_category": str(rng.choice(categories, p=weights / weights.sum())),
        "hour": int(rng.integers(0, 24)),
        "account_age_days": int(rng.choice([rng.integers(5, 60), rng.integers(60, 4000)], p=[0.15, 0.85])),
        "txn_count_last_hour": int(rng.poisson(1.5) + (rng.integers(3, 9) if rng.random() < 0.15 else 0)),
        "new_device": bool(rng.random() < 0.2),
    }


def generate_alerts(n_alerts: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return `n_alerts` alerted transactions with fired rules and a ground-truth label."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    idx = 0
    while len(rows) < n_alerts:
        idx += 1
        txn = _transaction(rng, idx)
        rules = _fired_rules(txn)
        if not rules:
            continue  # the monitoring system raised no alert for this transaction
        txn["rules_fired"] = ",".join(rules)
        txn["is_fraud"] = bool(rng.random() < _fraud_probability(txn))
        rows.append(txn)
    df = pd.DataFrame(rows)
    df["alert_id"] = [f"ALRT-{i + 1:04d}" for i in range(len(df))]
    return df
