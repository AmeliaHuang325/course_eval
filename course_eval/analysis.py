import re

import numpy as np
import pandas as pd


## Filter out and clean the data needed
def filter_max_attempts(df, drop_data=None, key=None):
    """Keep each learner's latest attempt and drop excluded learners.

    Canvas "Student Analysis" reports have one row per attempt. Learners are
    matched on the Canvas user ``id`` when present (names are not unique);
    otherwise on ``name``. ``drop_data`` is a collection of names to remove,
    e.g. test accounts.
    """
    if key is None:
        key = 'id' if 'id' in df.columns else 'name'

    # Keep the row(s) with the highest attempt number for each learner
    max_attempts = df.groupby(key)['attempt'].transform('max')
    filtered_df = df[df['attempt'] == max_attempts]

    # Drop rows where 'name' is in the drop_data list
    if drop_data:
        filtered_df = filtered_df[~filtered_df['name'].isin(drop_data)]

    return filtered_df.reset_index(drop=True)


## Calculate the Average
def avg_score(df, questions_no):
    # Calculate the score percentage and create a new column 'score_percentage'
    df['score_percentage'] = (df['score'] / questions_no) * 100

    return df


## Count the scored questions in a Canvas quiz report
def count_quiz_questions(df):
    """Number of question score columns ('1.0', '1.0.1', ...) in a quiz report."""
    return sum(1 for col in df.columns if re.fullmatch(r'\d+\.\d+(\.\d+)?', str(col)))


## Define the function to rename columns
def rename_columns_with_info_within_parentheses(df):
    """Rename question columns to the code in parentheses, e.g. '5728: (PRE_01) Which...' -> 'PRE_01'."""
    pattern = r'\((.*?)\)'  # Pattern to capture text within parentheses
    new_column_names = {}

    for col in df.columns:
        match = re.search(pattern, str(col))
        if match:
            new_column_names[col] = match.group(1)  # Extract and use the text within parentheses
        else:
            new_column_names[col] = col  # Use original column name if no match

    # Rename columns in place (callers rely on the DataFrame being updated)
    df.columns = [new_column_names[col] for col in df.columns]
    return df


## Define the false count and false rate for the tests
def false_counts_false_rate(df, test_questions_no):
    # Determine the naming convention by checking the existence of a specific column
    if '1.0' in df.columns:
        base_name = '1.0'
    elif '1' in df.columns:
        base_name = '1'
    else:
        raise ValueError("Column naming convention is not recognized.")

    # Lists to hold the count of zeros and false rates for each question
    question_labels = []
    false_counts = []
    false_rates = []

    # Iterate over each question column and count zeros
    for i in range(test_questions_no):
        # Formulate the column name based on the detected base name and question number
        column_name = f'{base_name}.{i}' if i > 0 else base_name

        if column_name in df.columns:
            count = (df[column_name] == 0).sum()
            rate = count / len(df)
            question_labels.append(f'q{i+1}')
            false_counts.append(count)
            false_rates.append(rate)
        else:
            raise ValueError(f"Column {column_name} does not exist in the DataFrame.")

    # Create a DataFrame from the counts and rates
    false_counts_df = pd.DataFrame({
        'false_counts': false_counts,
        'false_rate': false_rates
    }, index=question_labels)

    return false_counts_df


## Sentiment analysis
def _sentiment_analyzer():
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()


## Define the function to process the sentiment analysis
def process_sentiment(data, column, score_type):
    # Create a DataFrame from the specified column
    df = pd.DataFrame(data[column]).dropna()

    # Drop rows with 'na', 'NA', 'Na', etc., or only special symbols
    df = df[df[column].apply(lambda x: not (str(x).strip().lower() in ['na', 'n/a'] or not re.search('[a-zA-Z0-9]', str(x))))]

    # Calculate sentiment scores
    analyzer = _sentiment_analyzer()
    df[f'{score_type} sentiment score'] = df[column].apply(lambda x: analyzer.polarity_scores(str(x))[score_type])

    # Sort the DataFrame by sentiment scores
    return df.sort_values(by=f'{score_type} sentiment score', ascending=False)


## Calculate the 25th percentile of sentiment analysis
def plot_25th_percentile(data, column):
    # Ensure data is sorted
    sorted_data = data[column].dropna().sort_values()

    # Calculate the 25th percentile
    percentile_25 = np.percentile(sorted_data, 25)

    return percentile_25


## Calculate the Median Sentiment Score
def plot_median_sentiment_score(data, column):
    # Ensure data is sorted
    sorted_data = data[column].dropna().sort_values()

    # Calculate the median (50th percentile)
    median_score = np.percentile(sorted_data, 50)

    return median_score


## Calculate the 75th percentile of sentiment analysis
def plot_75th_percentile(data, column):
    # Ensure data is sorted
    sorted_data = data[column].dropna().sort_values()

    # Calculate the 75th percentile
    percentile_75 = np.percentile(sorted_data, 75)

    return percentile_75
