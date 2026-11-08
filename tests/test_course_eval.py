import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

import course_eval as ce


def quiz(prefix, rows, n_questions=3):
    """Minimal Canvas 'Student Analysis' quiz report: one row per (name, id, attempt, answers...)."""
    records = []
    for name, uid, attempt, answers in rows:
        rec = {"name": name, "id": uid, "section": "Demo", "attempt": attempt}
        for q, correct in enumerate(answers):
            rec[f"{1000 + q}: ({prefix}_{q + 1:02d}) Question {q + 1}"] = "a. Option" if correct else "b. Option"
            rec["1.0" if q == 0 else f"1.0.{q}"] = float(correct)
        rec["n correct"] = sum(answers)
        rec["n incorrect"] = n_questions - sum(answers)
        rec["score"] = float(sum(answers))
        records.append(rec)
    return pd.DataFrame(records)


@pytest.fixture
def pre():
    return quiz("PRE", [
        ("Learner 01", 1, 1, [1, 0, 0]),
        ("Learner 01", 1, 2, [1, 1, 0]),   # second attempt should win
        ("Learner 01", 2, 1, [0, 0, 1]),   # different learner, same display name
        ("Learner 03", 3, 1, [1, 1, 1]),
        ("Test Student", 4, 1, [0, 0, 0]),
    ])


@pytest.fixture
def post():
    return quiz("POST", [("Learner 01", 1, 1, [1, 1, 1]), ("Learner 03", 3, 1, [1, 1, 0])])


@pytest.fixture
def evaluation():
    return pd.DataFrame({
        "name": ["Learner 01", "Learner 02", "Learner 03", "Learner 04"],
        "id": [1, 2, 3, 4],
        "attempt": [1, 1, 1, 1],
        "EVAL_01": ["Little", "Basic", "Intermediate", "Not Applicable"],
        "EVAL_02": ["Intermediate", "Advanced", "Basic", "Advanced"],
        "EVAL_17": ["Great course, very helpful!", "Too long and confusing.", "n/a", "Okay overall."],
    })


def prepared(df):
    out = ce.filter_max_attempts(df, {"Test Student"})
    ce.rename_columns_with_info_within_parentheses(out)
    return ce.avg_score(out, ce.count_quiz_questions(out))


def test_filter_max_attempts_uses_id_not_name(pre):
    out = ce.filter_max_attempts(pre, {"Test Student"})
    assert sorted(out["id"]) == [1, 2, 3]             # both "Learner 01"s kept, test account dropped
    assert out.loc[out["id"] == 1, "attempt"].item() == 2


def test_count_and_rename(pre):
    assert ce.count_quiz_questions(pre) == 3
    out = ce.rename_columns_with_info_within_parentheses(pre.copy())
    assert {"PRE_01", "PRE_02", "PRE_03"} <= set(out.columns)


def test_avg_score_and_false_rates(pre):
    out = prepared(pre)
    assert out.loc[out["id"] == 3, "score_percentage"].item() == pytest.approx(100.0)
    rates = ce.false_counts_false_rate(out, 3)
    assert list(rates.index) == ["q1", "q2", "q3"]
    assert rates.loc["q3", "false_counts"] == 1        # only learner 1 missed q3 on their latest attempt


def test_sentiment_drops_na_and_sorts(evaluation):
    out = ce.process_sentiment(evaluation, "EVAL_17", "compound")
    assert len(out) == 3                               # "n/a" removed
    scores = out["compound sentiment score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0 > scores[-1]


def test_plots_return_figures_without_mutating_inputs(pre, post, evaluation):
    pre_p, post_p = prepared(pre), prepared(post)
    pre_before, eval_before = pre_p.copy(), evaluation.copy()
    combined = pd.concat([ce.false_counts_false_rate(pre_p, 3), ce.false_counts_false_rate(post_p, 3)],
                         axis=1, keys=["Pre_Test", "Post_Test"])
    sentiment = ce.process_sentiment(evaluation, "EVAL_17", "compound")

    figures = [
        ce.plot_test_score_distribution(pre_p, 3),
        ce.plot_test_false_rates(combined),
        ce.plot_pre_post_pie_charts("01", pre_p, post_p),
        ce.plot_funnel_diagram_no_program(10, pre_p, post_p, evaluation),
        ce.plot_participant_drop_between_tests_pre(pre_p, post_p),
        ce.plot_compound_sentiment_scores(sentiment, "compound sentiment score"),
        ce.plot_ksa_distribution(evaluation, "EVAL_01", "EVAL_02"),
        ce.plot_ksa_level_differences(evaluation, "EVAL_01", "EVAL_02"),
        ce.plot_individual_ksa_improvement(evaluation, "EVAL_01", "EVAL_02", 2),
    ]
    assert all(isinstance(f, Figure) for f in figures)
    pd.testing.assert_frame_equal(pre_p, pre_before)
    pd.testing.assert_frame_equal(evaluation, eval_before)


def test_ksa_level_differences_is_independent_and_labels_decreases(evaluation):
    fig = ce.plot_ksa_level_differences(evaluation, "EVAL_01", "EVAL_02")   # called first, raw capitalized values
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "1 level decrease" in labels and "2 level increase" in labels


def test_sankey_returns_plotly_figure():
    enroll = pd.DataFrame({"Canvas User Id": [1, 2, 3, 4], "Status": ["Completed", "Active", "Completed", "Dropped"]})
    course = enroll.iloc[:1]
    fig = ce.plot_participant_flow_sankey(enroll, course, enroll)
    assert isinstance(fig, go.Figure)
