# course_eval

A Python package for analyzing and visualizing **course evaluations from Canvas LMS exports**: pre/post test results, question-level difficulty, participant flow, self-assessed knowledge growth, and sentiment of open-ended feedback.

I built it to replace a manual, notebook-per-course workflow with reusable functions. The same analysis now runs on any course export in a few lines.

**See it in action:** the [Course Evaluation Dashboard](https://github.com/AmeliaHuang325/course-eval-dashboard) is an interactive Streamlit app built on this package.

## Install

```bash
pip install git+https://github.com/AmeliaHuang325/course_eval.git
```

## Input Data

The package works directly with Canvas exports:

| Export | Used for |
|---|---|
| Quiz **Student Analysis** report (pre-test, post-test) | Scores, attempts, per-question correctness |
| Survey **Student Analysis** report (course evaluation) | Self-rated level before/after, agreement statements, open-ended comments |
| **Canvas Catalog** enrollment export | Enrollment counts and status for the participant funnel |

The functions rely on Canvas's layout:
- **Question columns** are named like `5728: (PRE_01) Which of the following...`. `rename_columns_with_info_within_parentheses` shortens them to `PRE_01`.
- **Each question is followed by its points column** (`1.0`, `1.0.1`, ...), which the error-rate functions read.

## Quick Start

```python
import pandas as pd
import course_eval as ce

pre = pd.read_csv("Pre-test Quiz Student Analysis Report.csv")
post = pd.read_csv("Post-test Quiz Student Analysis Report.csv")
evaluation = pd.read_csv("Evaluation Survey Student Analysis Report.csv", keep_default_na=False, na_values="")

# Clean: latest attempt per learner, drop test accounts, short question names
pre, post, evaluation = (ce.filter_max_attempts(df, {"Test Student"}) for df in (pre, post, evaluation))
for df in (pre, post, evaluation):
    ce.rename_columns_with_info_within_parentheses(df)

# Score
n = ce.count_quiz_questions(pre)
pre, post = ce.avg_score(pre, n), ce.avg_score(post, n)
error_rates = pd.concat([ce.false_counts_false_rate(pre, n), ce.false_counts_false_rate(post, n)],
                        axis=1, keys=["Pre_Test", "Post_Test"])

# Visualize (every plot function returns its figure)
fig = ce.plot_test_false_rates(error_rates)
fig.savefig("error_rates.png", dpi=150)

# Sentiment of open-ended feedback (NLTK VADER)
comments = ce.process_sentiment(evaluation, "EVAL_17", "compound")
ce.plot_compound_sentiment_scores(comments, "compound sentiment score")
```

## Functions

**Analysis** (`course_eval.analysis`)

| Function | Purpose |
|---|---|
| `filter_max_attempts(df, drop_data=None)` | Keep each learner's latest attempt (matched on Canvas `id`) and drop excluded names |
| `rename_columns_with_info_within_parentheses(df)` | `"5728: (PRE_01) Question…"` → `"PRE_01"` |
| `count_quiz_questions(df)` | Number of scored questions in a quiz report |
| `avg_score(df, questions_no)` | Adds `score_percentage` |
| `false_counts_false_rate(df, questions_no)` | Incorrect-answer count and rate per question |
| `process_sentiment(df, column, score_type)` | VADER sentiment (`pos`, `neg`, `neu`, `compound`) per response, sorted |
| `plot_25th_percentile`, `plot_median_sentiment_score`, `plot_75th_percentile` | Sentiment percentiles |

**Plotting** (`course_eval.plotting`; matplotlib figures, plus a Plotly Sankey)

| Function | Chart |
|---|---|
| `plot_test_score_distribution` | Score histogram with average line |
| `plot_test_false_rates` | Pre vs. post incorrect-answer rate by question |
| `plot_pre_post_pie_charts` | Answer distribution for one question, before and after |
| `plot_funnel_diagram_no_program`, `plot_funnel_diagram_with_program` | Participant funnel |
| `plot_participant_flow_sankey` | Program → course → registration flow (Plotly) |
| `plot_participant_drop_between_tests_pre` | Pre-test scores of learners who stayed vs. left |
| `plot_compound_sentiment_scores` | Sentiment box plot with percentiles |
| `plot_ksa_distribution`, `plot_ksa_level_differences`, `plot_individual_ksa_improvement` | Self-rated knowledge/skills/abilities, before vs. after |

## Version 0.2

- Every plotting function returns its figure, so charts can be saved, embedded in reports, or shown in apps.
- **Fixes:**
  - Learners are matched on their Canvas user ID, so learners who share a name are no longer merged.
  - Plotting functions no longer modify the caller's DataFrames.
  - `plot_ksa_level_differences` works on its own, without `plot_ksa_distribution` running first.
  - Decreases are labeled "level decrease".
- **Compatibility:** works with pandas 3 and current matplotlib.
- Dependencies are declared, and the sentiment lexicon downloads automatically.
- New `count_quiz_questions()`, plus a pytest suite.

## Tests

```bash
pip install -e ".[test]"
pytest
```
