import json
import unittest

import pandas as pd

from triage.backends import BaselineBackend, _parse_decision
from triage.data import generate_alerts
from triage.evaluate import evaluate
from triage.prompt import describe_alert


class DataTests(unittest.TestCase):
    def test_generation_is_reproducible_and_every_alert_fired_a_rule(self):
        a = generate_alerts(50, seed=1)
        b = generate_alerts(50, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 50)
        self.assertTrue((a["rules_fired"].str.len() > 0).all())

    def test_most_alerts_are_false_positives_like_real_rule_systems(self):
        df = generate_alerts(500, seed=7)
        self.assertTrue(0.05 < df["is_fraud"].mean() < 0.5)

    def test_prompt_never_leaks_the_label(self):
        alert = generate_alerts(5, seed=3).iloc[0].to_dict()
        self.assertNotIn("is_fraud", describe_alert(alert))


class BackendTests(unittest.TestCase):
    def test_baseline_escalates_when_enough_rules_fire(self):
        backend = BaselineBackend(threshold=2)
        self.assertEqual(backend.triage({"rules_fired": "HIGH_AMOUNT"}).decision, "close")
        self.assertEqual(backend.triage({"rules_fired": "HIGH_AMOUNT,NEW_DEVICE"}).decision, "escalate")

    def test_parse_decision_validates_and_clamps(self):
        d = _parse_decision(json.dumps({"decision": "Escalate", "risk_score": 140, "reason": " x "}))
        self.assertEqual((d.decision, d.risk_score, d.reason), ("escalate", 100, "x"))
        with self.assertRaises(ValueError):
            _parse_decision(json.dumps({"decision": "maybe", "risk_score": 1, "reason": ""}))


class EvaluateTests(unittest.TestCase):
    def test_metrics_on_a_small_known_case(self):
        results = pd.DataFrame({
            "is_fraud": [True, True, False, False],
            "decision": ["escalate", "close", "escalate", "close"],
            "risk_score": [90, 40, 60, 10],
        })
        m = evaluate(results)
        self.assertEqual((m["true_positives"], m["missed_fraud"], m["false_positives"]), (1, 1, 1))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["workload_reduction"], 0.5)


if __name__ == "__main__":
    unittest.main()
