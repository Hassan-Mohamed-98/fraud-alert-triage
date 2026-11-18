# Fraud Alert Triage: can an LLM reduce false positives from rule-based monitoring?

Rule-based transaction monitoring raises many alerts, and most of them are false positives
that an investigator still has to open. This project tests whether a large language model
can triage those alerts, escalating the suspicious ones and closing the rest, and **measures**
the result against a rule baseline instead of trusting the model's judgement.

All data is **synthetic**. No real customer or bank data is used.

## How it works

1. **Synthetic alerts** (`triage/data.py`): customer card transactions are generated and five
   simple monitoring rules fire alerts (high amount, foreign transaction, velocity, new device,
   risky merchant). Whether an alerted transaction is really fraud is decided by a hidden, noisy
   combination of risk factors that no single rule sees, so most alerts are false positives,
   as in real monitoring.
2. **Triage backends** (`triage/backends.py`), all behind the same interface:
   - `baseline`: escalate when two or more rules fired (the bar to beat)
   - `ollama:<model>`: a free LLM running locally via [Ollama](https://ollama.com)
   - `claude[:<model>]`: Claude via the Anthropic API (needs an API key)

   The LLM gets the alert as text (`triage/prompt.py`) and must answer in a fixed JSON schema:
   a decision (`escalate` / `close`), a 0-100 risk score and a short reason an investigator can check.
3. **Evaluation** (`triage/evaluate.py`): precision, recall, F1, missed fraud, false positives,
   investigator workload reduction and ROC AUC of the risk score, all against the hidden labels.

## Quick start

```bash
pip install -r requirements.txt
python run.py generate --n 300          # creates data/alerts.csv
python run.py triage --backend baseline # scores the rule baseline
```

### Run a local LLM (free, no API key)

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b
python run.py triage --backend ollama:llama3.2:3b --limit 100
python run.py compare
```

### Run Claude (optional)

```bash
pip install anthropic
set ANTHROPIC_API_KEY=...               # PowerShell: $env:ANTHROPIC_API_KEY="..."
python run.py triage --backend claude --limit 100
```

## Results

| Backend | Alerts | Precision | Recall | F1 | Missed fraud | False positives | Workload reduction |
|---|---|---|---|---|---|---|---|
| baseline (2+ rules) | 300 | 0.35 | 0.68 | 0.47 | 24 | 91 | 53% |
| LLM | *to be added after running* | | | | | | |

Per-alert decisions and reasons are saved in `results/<backend>.csv`, metrics in `results/<backend>.json`.

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Limitations

- The ground truth is synthetic, so results show how the method behaves, not how a model
  would perform on a real bank's alerts.
- Small local models are much weaker than hosted models; compare several before drawing conclusions.
- A production setup would need human review of escalation decisions, audit logging and
  model monitoring, which is out of scope here.

## Author

Hassan Kaka, developed with the help of Claude Code.
