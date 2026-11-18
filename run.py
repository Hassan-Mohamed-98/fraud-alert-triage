"""Command-line entry point.

Examples:
    python run.py generate --n 300
    python run.py triage --backend baseline
    python run.py triage --backend ollama:llama3.2:3b --limit 50
    python run.py triage --backend claude --limit 50
    python run.py compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from triage.backends import make_backend
from triage.data import generate_alerts
from triage.evaluate import evaluate, format_report

DATA = Path("data/alerts.csv")
RESULTS = Path("results")


def cmd_generate(args: argparse.Namespace) -> None:
    df = generate_alerts(args.n, args.seed)
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False)
    print(f"Wrote {len(df)} alerts to {DATA} ({df['is_fraud'].mean():.0%} fraud).")


def _load_alerts(limit: int | None) -> pd.DataFrame:
    if not DATA.exists():
        sys.exit(f"{DATA} not found. Run `python run.py generate` first.")
    df = pd.read_csv(DATA)
    return df.head(limit) if limit else df


def cmd_triage(args: argparse.Namespace) -> None:
    alerts = _load_alerts(args.limit)
    backend = make_backend(args.backend)
    rows = []
    started = time.time()
    for i, alert in enumerate(alerts.to_dict("records"), start=1):
        try:
            d = backend.triage(alert)
            rows.append({"alert_id": alert["alert_id"], "is_fraud": alert["is_fraud"],
                         "decision": d.decision, "risk_score": d.risk_score, "reason": d.reason})
        except Exception as exc:  # keep going; failures are reported, not hidden
            rows.append({"alert_id": alert["alert_id"], "is_fraud": alert["is_fraud"],
                         "decision": "escalate", "risk_score": 50, "reason": f"ERROR: {exc}"})
        if i % 25 == 0 or i == len(alerts):
            print(f"  {i}/{len(alerts)} alerts", flush=True)
    results = pd.DataFrame(rows)
    errors = results["reason"].str.startswith("ERROR").sum()
    if errors == len(results):
        sys.exit(f"Every alert failed, nothing saved. First error: {results['reason'].iloc[0]}")
    metrics = evaluate(results)
    metrics.update({"backend": backend.name, "errors": int(errors),
                    "seconds": round(time.time() - started, 1)})

    RESULTS.mkdir(exist_ok=True)
    slug = backend.name.replace(":", "_").replace("/", "_")
    results.to_csv(RESULTS / f"{slug}.csv", index=False)
    (RESULTS / f"{slug}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(format_report(backend.name, metrics))
    if errors:
        print(f"WARNING: {errors} alert(s) failed and were escalated by default; see the CSV.")


def cmd_compare(_: argparse.Namespace) -> None:
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        sys.exit("No results yet. Run `python run.py triage --backend ...` first.")
    table = pd.DataFrame([json.loads(f.read_text(encoding="utf-8")) for f in files])
    cols = ["backend", "alerts", "precision", "recall", "f1", "missed_fraud",
            "false_positives", "workload_reduction", "errors"]
    print(table[[c for c in cols if c in table]].round(2).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud alert triage with a rule baseline and LLMs.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create the synthetic alert dataset")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("triage", help="run one backend over the alerts and score it")
    t.add_argument("--backend", default="baseline", help="baseline | ollama[:model] | claude[:model]")
    t.add_argument("--limit", type=int, default=None, help="only the first N alerts")
    t.set_defaults(func=cmd_triage)

    c = sub.add_parser("compare", help="compare all scored backends")
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
