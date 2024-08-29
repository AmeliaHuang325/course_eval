import pandas as pd
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

## Filter out and clean the data needed
def filter_max_attempts(df, drop_data):
    # Group by 'name' and find the maximum 'attempt' for each group
    max_attempts = df.groupby('name')['attempt'].transform(max)

    # Filter the DataFrame to keep only rows where 'attempt' equals the maximum for that group
    filtered_df = df[df['attempt'] == max_attempts]

    # Drop rows where 'name' is in the drop_data list
    filtered_df = filtered_df[~filtered_df['name'].isin(drop_data)]

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df

## Calculate the Average
def avg_score(df, questions_no):
    # Calculate the score percentage and create a new column 'score_percentage'
    df['score_percentage'] = (df['score'] / questions_no) * 100

    return df

## Define the function to rename columns
def rename_columns_with_info_within_parentheses(df):
    pattern = r'\((.*?)\)'  # Pattern to capture text within parentheses
    new_column_names = {}

    for col in df.columns:
        match = re.search(pattern, col)
        if match:
            new_column_names[col] = match.group(1)  # Extract and use the text within parentheses
        else:
            new_column_names[col] = col  # Use original column name if no match

    # Rename columns in the DataFrame
    df.rename(columns=new_column_names, inplace=True)

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
        if base_name == '1.0':
            column_name = f'{base_name}.{i}' if i > 0 else '1.0'
        elif base_name == '1':
            column_name = f'{base_name}.{i}' if i > 0 else '1'

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

## Define the function to process the sentiment analysis
def process_sentiment(data, column, score_type):
    # Create a DataFrame from the specified column
    df = pd.DataFrame(data[column]).dropna()
    
    # Drop rows with 'na', 'NA', 'Na', etc., or only special symbols
    df = df[df[column].apply(lambda x: not (str(x).strip().lower() in ['na', 'n/a'] or not re.search('[a-zA-Z0-9]', str(x))))]

    # Calculate sentiment scores
    analyzer = SentimentIntensityAnalyzer()
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

## Calculate the 75th percentile of sentiment analysis
def plot_75th_percentile(data, column):
    # Ensure data is sorted
    sorted_data = data[column].dropna().sort_values()
    
    # Calculate the 75th percentile
    percentile_75 = np.percentile(sorted_data, 75)
    
    return percentile_75

