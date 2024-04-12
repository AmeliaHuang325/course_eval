import pandas as pd
import re

# Filter out and clean the data needed
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

def avg_score(df, questions_no):
    # Calculate the score percentage and create a new column 'score_percentage'
    df['score_percentage'] = (df['score'] / questions_no) * 100

    return df

# Define the function to rename columns
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
