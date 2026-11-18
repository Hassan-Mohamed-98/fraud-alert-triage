"""Score triage decisions against the ground truth, the way an operations team would."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(results: pd.DataFrame) -> dict:
    """`results` needs columns is_fraud (bool), decision ('escalate'/'close') and risk_score (0-100)."""
    y_true = results["is_fraud"].astype(bool)
    y_pred = results["decision"].eq("escalate")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    metrics = {
        "alerts": int(len(results)),
        "fraud_cases": int(y_true.sum()),
        "escalated": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "missed_fraud": int(fn),
        "correctly_closed": int(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # Share of alerts an investigator no longer has to open
        "workload_reduction": float(1 - y_pred.mean()),
    }
    if y_true.nunique() == 2:
        metrics["roc_auc_of_risk_score"] = float(roc_auc_score(y_true, results["risk_score"]))
    return metrics


def format_report(name: str, metrics: dict) -> str:
    lines = [
        f"== {name} ==",
        f"alerts: {metrics['alerts']}  (fraud: {metrics['fraud_cases']}, escalated: {metrics['escalated']})",
        f"precision: {metrics['precision']:.2f}   recall: {metrics['recall']:.2f}   F1: {metrics['f1']:.2f}",
        f"missed fraud: {metrics['missed_fraud']}   false positives: {metrics['false_positives']}",
        f"investigator workload reduction: {metrics['workload_reduction']:.0%}",
    ]
    if "roc_auc_of_risk_score" in metrics:
        lines.append(f"ROC AUC of risk score: {metrics['roc_auc_of_risk_score']:.2f}")
    return "\n".join(lines)
