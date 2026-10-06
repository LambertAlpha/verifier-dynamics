"""E010 real-grader reimplementations (lm-evaluation-harness GSM8K filters, loose extraction)."""

from vdyn.e007 import graders as gr


def test_lm_eval_strict_needs_the_hash_format():
    assert gr.lm_eval_strict("so ... #### 1,234", "1234") == 1.0
    assert gr.lm_eval_strict("the answer is 1234", "1234") == 0.0


def test_lm_eval_flexible_takes_the_last_number():
    assert gr.lm_eval_flexible("3 + 4 = 7, so $72.", "72") == 1.0
    assert gr.lm_eval_flexible("72 apples, then 5", "72") == 0.0
    assert gr.lm_eval_flexible("no digits", "72") == 0.0


def test_anywhere_accepts_the_reference_anywhere():
    assert gr.anywhere("72 - 1 = 71", "72") == 1.0
    assert gr.anywhere("71", "72") == 0.0
