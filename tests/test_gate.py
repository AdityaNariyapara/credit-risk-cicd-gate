"""Quality-gate logic tests (Task 6)."""
from credit_risk.gate import evaluate_gate


BASELINE = {"metrics": {"average_precision": 0.60}}


def test_pass_when_equal():
    r = evaluate_gate({"average_precision": 0.60}, BASELINE, "average_precision", 0.0)
    assert r.passed


def test_pass_within_tolerance():
    r = evaluate_gate({"average_precision": 0.595}, BASELINE, "average_precision", 0.01)
    assert r.passed


def test_fail_when_worse_than_tolerance():
    r = evaluate_gate({"average_precision": 0.58}, BASELINE, "average_precision", 0.01)
    assert not r.passed


def test_pass_when_better():
    r = evaluate_gate({"average_precision": 0.70}, BASELINE, "average_precision", 0.0)
    assert r.passed


def test_no_baseline_accepts():
    r = evaluate_gate({"average_precision": 0.4}, None, "average_precision", 0.0)
    assert r.passed
    assert r.baseline is None
