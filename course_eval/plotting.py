import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import plotly.graph_objects as go

## Function to plot pre-test and post-test score distribution

def plot_test_score_distribution(filtered_test, test_questions_no):
    # Calculate the required statistics
    average_score = filtered_test['score_percentage'].mean()
    high_score = filtered_test['score_percentage'].max()
    low_score = filtered_test['score_percentage'].min()
    standard_deviation = filtered_test['score_percentage'].std()

    # Set the color for the bars
    bar_color = "#3498db"

    # Custom dash pattern: longer dash, longer gap
    dash_pattern = [5, 3]

    # set opacity value
    alpha_value = 0.8

    # Using seaborn for a nicer looking plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    # Calculate the score increment per question
    score_increment = 100 / test_questions_no

    # Find the nearest multiple of score_increment that is less than or equal to the bin width
    bin_start = score_increment * (2.5 // score_increment)

    # Shift the starting point of the bins to the left by half the width of a bar
    bin_start_shifted = bin_start - 1.25

    # Adjust bin edges based on the number of questions, shifted to center-align the bars
    bins = np.arange(bin_start_shifted, 100 + score_increment, 2.5)

    # Plot the histogram
    hist_plot = sns.histplot(filtered_test['score_percentage'], bins=bins, kde=False, color=bar_color, alpha=alpha_value)

    # Loop through the bars and add text annotations
    for p in hist_plot.patches:
        height = p.get_height()
        if height > 0:
            plt.text(p.get_x() + p.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom', alpha=alpha_value)


    # Remove the grid
    hist_plot.grid(False)

    # Limit x-axis
    plt.xlim(0, 105)

    # Adjust y-axis tick labels to remove the zero
    y_ticks = plt.gca().get_yticks()
    y_ticks = y_ticks[y_ticks != 0]
    plt.gca().set_yticks(y_ticks)

    # Hide top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Set bottom (x-axis) spine to dashed and match bar color
    plt.gca().spines['bottom'].set_color(bar_color)
    plt.gca().spines['bottom'].set_linestyle((0, dash_pattern))
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().spines['bottom'].set_alpha(alpha_value)

    # Adjust y-axis spine opacity
    plt.gca().spines['left'].set_alpha(alpha_value)

    # Add a vertical line for the average score
    plt.axvline(average_score, color='red', linestyle=(0, dash_pattern), linewidth=1, alpha=alpha_value)

    # Add text for average score next to the vertical line
    plt.text(average_score + 1, plt.ylim()[1] * 0.95, f'Avg: {average_score:.2f}%', color='red', verticalalignment='top', alpha=alpha_value, fontsize=9)

    # Set titles and labels
    plt.xlabel('Score', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')

    # Calculate position for the info text
    info_text_y = plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0]) * 0.2

    # Add text for high score, low score, and std dev
    plt.text(plt.xlim()[1] / 2, info_text_y, f'Lowest Score: {low_score}%, Highest Score: {high_score}%', horizontalalignment='center', alpha=alpha_value, fontsize=9)

    return plt

## Generate a diagram for the pre-test and post-test false rate by question
def plot_test_false_rates(data, text_size=10):
    # Set the positions of the bars on the x-axis
    ind = range(len(data))  # the x locations for the groups

    # Bar width
    width = 0.35

    # Bar colors
    bar_color_pre = "#80c57e"
    bar_color_post = "#da6aae"

    # Dash line pattern
    dash_pattern_test = [5, 3]

    # Opacity
    alpha_value_test = 0.8

    # Create plotting area
    fig, ax = plt.subplots(figsize=(10, 5)) 
    rects1 = ax.bar(ind, data['Pre_Test']['false_rate'], width, label='Pre-Test', color=bar_color_pre)
    rects2 = ax.bar([i + width for i in ind], data['Post_Test']['false_rate'], width, label='Post-Test', color=bar_color_post)

    # Remove the grid
    ax.grid(False)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set bottom (x-axis) spine to dashed and match bar color
    ax.spines['bottom'].set_color(bar_color_post)
    ax.spines['bottom'].set_linestyle((0, dash_pattern_test))
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['bottom'].set_alpha(alpha_value_test)

    # Set labels, title and custom x-axis tick labels
    ax.set_xlabel('Questions No.', fontweight='bold')
    ax.set_ylabel('False Rate', fontweight='bold')
    ax.set_title('Pre-test and Post-test False Rate by Question', fontweight='bold')
    ax.set_xticks([i + width / 2 for i in ind])
    ax.set_xticklabels(data.index)

    # Move the legend to the right side of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Function to annotate bars
    def annotate_bars(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=text_size)

    # Use the function to annotate bars
    annotate_bars(rects1, ax)
    annotate_bars(rects2, ax)

    plt.tight_layout()  # Adjust layout to fit everything

## Functions to create pie charts for specific question
# Function to replace labels
def replace_label(label):
    if isinstance(label, str):
        label = label.lower()
        if label.startswith('a.'):
            return 'Answer 1'
        elif label.startswith('b.'):
            return 'Answer 2'
        elif label.startswith('c.'):
            return 'Answer 3'
        elif label.startswith('d.'):
            return 'Answer 4'
        else:
            return label.title()
    else:
        return 'NaN'

# Function to create the pie chart
def create_pie_chart(data_series, title, color_scheme, ax):
    # Count the occurrences of each unique value
    counts = data_series.value_counts()

    # Sort the index of the answers
    sorted_index = ['Answer 1', 'Answer 2', 'Answer 3', 'Answer 4']
    sorted_counts = counts.reindex(sorted_index).dropna()

    # Plotting the pie chart
    patches, texts, autotexts = ax.pie(sorted_counts, colors=color_scheme, autopct='%1.1f%%', startangle=90)

    # Change the font size of the autopct values
    for autotext in autotexts:
        autotext.set_fontsize(10)
    
    # Adding dotted circles
    circle = plt.Circle((0, 0), 1.1, color='black', fill=False, linestyle='dotted', linewidth=0.7, alpha=0.5)
    ax.add_artist(circle)

    # Adding custom annotations
    size = 1.1
    ax.text(0, size, '  0.00 | 1.00', ha='center', va='center', fontsize=10)
    ax.text(0, -size, '0.25', ha='center', va='center', fontsize=10)
    ax.text(-size, 0, '0.50', ha='center', va='center', fontsize=10)
    ax.text(size, 0, '0.75', ha='center', va='center', fontsize=10)
    
    # Adding a legend on the right side of the pie chart
    ax.legend(patches, sorted_counts.index, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=12)

# Function to plot the pre-test and post-test pie charts
def plot_pre_post_pie_charts(column_suffix, filtered_pre_test, filtered_post_test):
    # Creating a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

     # Adjust the space between the two pie charts and the overall layout
    plt.subplots_adjust(left=0, right=0.95, wspace=0.2)  

    # Process the data for pre-test and post-test
    pre_column = 'PRE_' + column_suffix
    post_column = 'POST_' + column_suffix
    filtered_pre_test[pre_column] = filtered_pre_test[pre_column].apply(replace_label)
    filtered_post_test[post_column] = filtered_post_test[post_column].apply(replace_label)

    # Define color schemes
    colors_pre = ['#fffecf', '#c5e59e', '#80c57e', '#338347']
    colors_post = ['#f1eef7', '#d6b6d7', '#da6aae', '#c82c56']

    # Plot for pre-test
    create_pie_chart(filtered_pre_test[pre_column], f'Pre-Test Q{column_suffix}', colors_pre, axs[0])

    # Plot for post-test
    create_pie_chart(filtered_post_test[post_column], f'Post-Test Q{column_suffix}', colors_post, axs[1])

## Generate Funnel Diagram without Enrollment in the program
def plot_funnel_diagram_no_program(num_enrollment_course, filtered_pre_test, filtered_post_test, filtered_eval_info):
    # Calculate the number of rows for each DataFrame
    num_pre_test_rows = len(filtered_pre_test)
    num_post_test_rows = len(filtered_post_test)
    num_eval_rows = len(filtered_eval_info)

    # Values and labels for the plot
    values = [num_enrollment_course, num_pre_test_rows, num_post_test_rows, num_eval_rows]
    labels = ['Enrollment', 'Pre-Test', 'Post-Test', 'Evaluation']
    colors = ['#3498db', '#80c57e', '#E67E22', '#F1C40F']
    percentages = [value / values[0] * 100 for value in values]

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each stage of the funnel
    for i, (label, value, percentage) in enumerate(zip(labels, values, percentages)):
        # Calculate the left position to center the bars
        left = (max(percentages) - percentage) / 2
        # Plot the bar with the left position
        ax.barh(i, percentage, color=colors[i], align='center', left=left)
        # Place label text at the top left corner of the bar
        ax.text(left + 1, i - 0.25, label, va='center', ha='left', color='black', fontsize=12)
        # Center the percentage and row number text in the bar
        ax.text(left + percentage / 2, i, f'{percentage:.0f}%   ({value} Participants)', va='center', ha='center', color='Black', fontsize=12)

    # Configuration to clean up and format the plot
    ax.spines.values().__class__(map(lambda spine: spine.set_visible(False), ax.spines.values()))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.set_xlim(0, 100)
    ax.set_yticklabels([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set labels and title
    ax.set_xlabel('Percentage of Total Enrollment', labelpad=20, weight='bold', size=12)

    # Invert the y-axis so the largest bar is on top
    ax.invert_yaxis()

    # Show the plot
    plt.tight_layout()

## Generate Funnel Diagram with Enrollment in the program
def plot_funnel_diagram_with_program(num_enrollment_program, num_enrollment_course, filtered_pre_test, filtered_post_test, filtered_eval_info):
    # Calculate the number of rows for each DataFrame
    num_pre_test_rows = len(filtered_pre_test)
    num_post_test_rows = len(filtered_post_test)
    num_eval_rows = len(filtered_eval_info)

    # Values and labels for the plot
    values = [num_enrollment_program, num_enrollment_course, num_pre_test_rows, num_post_test_rows, num_eval_rows]
    labels = ['Enrollment in the Program', 'Enrollment in the Course', 'Pre-Test', 'Post-Test', 'Evaluation']
    colors = ['#480D46', '#3498db', '#80c57e', '#E67E22', '#F1C40F']
    percentages = [value / values[0] * 100 for value in values]

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each stage of the funnel
    for i, (label, value, percentage) in enumerate(zip(labels, values, percentages)):
        # Calculate the left position to center the bars
        left = (max(percentages) - percentage) / 2
        # Plot the bar with the left position
        ax.barh(i, percentage, color=colors[i], align='center', alpha=0.6, left=left)
        # Place label text at the top left corner of the bar
        ax.text(left + 1, i - 0.25, label, va='center', ha='left', color='black', fontsize=18)
        # Center the percentage and row number text in the bar
        ax.text(left + percentage / 2, i, f'{percentage:.0f}%   ({value} Participants)', va='center', ha='center', color='Black', fontsize=18)

    # Remove spines and ticks, and set the x-axis formatter
    ax.spines.values().__class__(map(lambda spine: spine.set_visible(False), ax.spines.values()))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.set_xlim(0, 100)
    ax.set_yticklabels([])

    # Remove the bar outlines (spines)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set labels and title
    ax.set_xlabel('Percentage of Total Enrollment', labelpad=20, weight='bold', size=12)

    # Invert the y-axis so the largest bar is on top
    ax.invert_yaxis()

    # Show the plot
    plt.tight_layout()

## Generate Sankey Diagram for the Participant Flow
def clean_course_status(df):
    df_cleaned = df[df['Status'] != 'Dropped']
    df_cleaned = df_cleaned.drop_duplicates(subset=['Canvas User Id'], keep='first')
    return df_cleaned

def plot_participant_flow_sankey(enrollment_program, enrollment_course, total_registration):
    # Clean data
    enrollment_program_2 = clean_course_status(enrollment_program)
    enrollment_course_2 = clean_course_status(enrollment_course)
    total_registration_2 = clean_course_status(total_registration)

    # Analyze flows
    participants_not_in_course = enrollment_program_2[~enrollment_program_2['Canvas User Id'].isin(enrollment_course_2['Canvas User Id'])]
    participants_not_in_course_in_reg = participants_not_in_course[participants_not_in_course['Canvas User Id'].isin(total_registration_2['Canvas User Id'])]
    participants_not_in_course_no_reg = participants_not_in_course[~participants_not_in_course['Canvas User Id'].isin(total_registration_2['Canvas User Id'])]

    registration_not_in_course = total_registration_2[total_registration_2['Canvas User Id'].isin(participants_not_in_course['Canvas User Id'])]
    registration_not_in_course_completed = registration_not_in_course[registration_not_in_course['Status'] == 'Completed']
    registration_not_in_course_active = registration_not_in_course[registration_not_in_course['Status'] == 'Active']

    # Create Sankey diagram
    node_flows = [len(participants_not_in_course), len(participants_not_in_course_no_reg), len(participants_not_in_course_in_reg), len(registration_not_in_course_completed), len(registration_not_in_course_active)]
    node_labels = ["Participants not in Course", "Not in Registration", "In Registration", "Finished the Registration", "Not Finished the Registration"]
    updated_labels = [f"{label} ({flow})" for label, flow in zip(node_labels, node_flows)]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=updated_labels,
            color=['#480D46', '#3498db', '#80c57e', '#E67E22', '#F1C40F']
        ),
        link=dict(
            source=[0, 0, 2, 2],  # Indices correlate to 'node_labels'
            target=[1, 2, 3, 4],
            value=[len(participants_not_in_course_no_reg), len(participants_not_in_course_in_reg), len(registration_not_in_course_completed), len(registration_not_in_course_active)]
        ))])

    fig.update_layout(
        title_text="Participant Flow Analysis",
        font_size=17,
        height=600
    )
    fig.show()

## Generate a diagram to see whether the pre-test score determines the drop between pre-test and post-test
def plot_participant_drop_between_tests_pre(filtered_pre_test, filtered_post_test):
    # Analyze participants who dropped and who took both tests
    participants_drop_between_test = filtered_pre_test[~filtered_pre_test['id'].isin(filtered_post_test['id'])]
    participants_both_test_pre = filtered_pre_test[filtered_pre_test['id'].isin(filtered_post_test['id'])]

    # Count scores
    count_drop = participants_drop_between_test['score_percentage'].value_counts().sort_index()
    count_both = participants_both_test_pre['score_percentage'].value_counts().sort_index()

    # Get all score percentages to align the bars in the plot
    all_scores = sorted(set(count_drop.index) | set(count_both.index))
    count_drop = count_drop.reindex(all_scores, fill_value=0)
    count_both = count_both.reindex(all_scores, fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 2.5

    # Create stacked bars
    drop_bars = ax.bar(count_drop.index, count_drop.values, width=bar_width, label='Drop Between the Tests', color='#BF5254')
    both_bars = ax.bar(count_both.index, count_both.values, bottom=count_drop.values, width=bar_width, label='Attend Both Tests', color='#94AACD')

    ax.set_xlabel('Pre-test Score', fontweight='bold')
    ax.set_ylabel('Number of Participants', fontweight='bold')
    ax.set_title('Participants by Pre-Test Score Percentage', fontweight='bold')
    plt.xticks(all_scores)
    plt.legend()

    # Hide the grid and the top and right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adding numbers on the bars
    for bar, value in zip(drop_bars, count_drop.values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height()-0.15, str(value), ha='center', va='top', color='white')

    for bar, base_value, value in zip(both_bars, count_drop.values, count_both.values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, base_value + value-0.15, str(value), ha='center', va='top', color='white')

    # Adding total count in black at the top of each bar
    for i, total in enumerate(count_drop.values + count_both.values):
        ax.text(all_scores[i], total + 0.2, str(total), ha='center', va='bottom', color='black')

    # Make the y-axis labels integers
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

## Generate a diagram for Compound Sentiment Score
def plot_compound_sentiment_scores(data, column):
    # Create the figure and axis object
    fig, ax = plt.subplots(figsize=(10, 2))

    # Create the boxplot with whiskers set to 5th and 95th percentiles
    flierprops = dict(marker='o', markerfacecolor='blue', markersize=2, linestyle='none', alpha=0.4)
    medianprops = dict(color='red', linewidth=2)

    # Ensure data is sorted for plotting
    sorted_data = data[column].dropna().sort_values()

    # Create the boxplot with customized flierprops and medianprops
    boxplot_elements = ax.boxplot(sorted_data, vert=False, whis=[5, 95],
                                  showfliers=True, patch_artist=True, flierprops=flierprops, medianprops=medianprops)

    # Calculate the 25th and 75th percentiles
    percentile_25 = np.percentile(sorted_data, 25)
    percentile_75 = np.percentile(sorted_data, 75)

    # Color the boxes and add the green rectangle
    for box in boxplot_elements['boxes']:
        median_value = boxplot_elements['medians'][0].get_xdata()[0]
        path = box.get_path()
        vertices = path.vertices
        y_bottom = vertices[0][1]
        height = vertices[2][1] - y_bottom

        # Yellow rectangle from 25th percentile to median
        yellow_width = median_value - percentile_25
        ax.add_patch(plt.Rectangle((percentile_25, y_bottom), yellow_width, height, 
                                   facecolor='#E67E22', edgecolor='black', zorder=2, clip_on=False))

        # Green rectangle from median to 75th percentile
        green_width = percentile_75 - median_value
        ax.add_patch(plt.Rectangle((median_value, y_bottom), green_width, height, 
                                   facecolor='#F1C40F', edgecolor='black', zorder=2, clip_on=False))

    # Plotting all data points as blue dots
    all_data = sorted_data
    ax.plot(all_data, np.zeros_like(all_data) + 1, 'b.', alpha=0.4)

    # Annotating the median value with a vertical arrow
    ax.annotate(f'Median = {median_value:.2f}', xy=(median_value, 1.1), xytext=(median_value, 1.3),
                arrowprops=dict(facecolor='black', arrowstyle="->", lw=1), fontsize=10, ha='center')

    # Set x-axis range from -1 to 1
    ax.set_xlim(-1, 1)

    # Only show bottom spine (x-axis line)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('black')

    # Remove y-axis tick marks
    ax.set_yticks([])

    # Set x-axis label
    ax.set_xlabel('Scores')

    # Tight layout for better spacing
    plt.tight_layout()


## Generate a diagram for KSA distribution
def plot_ksa_distribution(filtered_eval_info, eval_col_before, eval_col_after, na_values=None):
    # Define the custom order and extract keys for sorting
    custom_order = ['Advanced (5)', 'Intermediate (4)', 'Basic (3)', 'Little (2)', 'None (1)']
    custom_order_keys = [label.split(' ')[0].lower() for label in custom_order]

    # Convert columns to string, lowercase, and replace specified values with np.nan
    if na_values is None:
        na_values = ['nan', 'na', 'not applicable']
    
    filtered_eval_info[eval_col_before] = filtered_eval_info[eval_col_before].astype(str).str.lower().replace(na_values, np.nan)
    filtered_eval_info[eval_col_after] = filtered_eval_info[eval_col_after].astype(str).str.lower().replace(na_values, np.nan)

    # Copy to a new DataFrame and drop rows with NaNs in the specified EVAL columns
    filtered_eval_info_2 = filtered_eval_info.dropna(subset=[eval_col_before, eval_col_after])

    # Calculate value counts with reindexing to maintain order
    value_counts_before = filtered_eval_info_2[eval_col_before].value_counts().reindex(custom_order_keys, fill_value=0)
    value_counts_after = filtered_eval_info_2[eval_col_after].value_counts().reindex(custom_order_keys, fill_value=0)

    # Create a DataFrame with the results
    ksa_result_df = pd.DataFrame({'Before': value_counts_before, 'After': value_counts_after})
    ksa_result_df.index = custom_order  # Rename the index using the full custom_order

    # Plot the data as a side-by-side bar chart
    width = 0.35
    x = np.arange(len(custom_order))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ksa_result_df['Before'], width, color='#9dcaea', label='Before', alpha=0.8)
    bars2 = ax.bar(x + width/2, ksa_result_df['After'], width, color='#2980B9', label='After')

    ax.set_ylabel('Number of Participants', fontweight='bold')
    ax.set_title('Distribution of Knowledge/Skills/Abilities Level Before & After the Course', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ksa_result_df.index)
    ax.legend()
    ax.grid(False)

    # Add labels to the bars
    for bar, count in zip(bars1, ksa_result_df['Before']):
        ax.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha='center', va='bottom')

    for bar, count in zip(bars2, ksa_result_df['After']):
        ax.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha='center', va='bottom')

## Generate a diagram for KSA level difference
def plot_ksa_level_differences(filtered_eval_info, eval_col_before, eval_col_after):
    # Create a dictionary to map values to labels
    value_to_label = {
        'none': '1',
        'little': '2',
        'basic': '3',
        'intermediate': '4',
        'advanced': '5'
    }

    # Copy the input DataFrame to avoid modifying the original data
    data_mapped = filtered_eval_info.copy()

    # Map the values to labels and convert to numeric
    for col in [eval_col_before, eval_col_after]:
        data_mapped[col] = pd.to_numeric(data_mapped[col].map(value_to_label))

    # Calculate the difference and store it in a new column
    data_mapped['Difference'] = data_mapped[eval_col_after] - data_mapped[eval_col_before]

    # Prepare the data for plotting
    ksa_diff_df = data_mapped['Difference'].value_counts().sort_index().reset_index()
    ksa_diff_df.columns = ['Difference', 'Number of Participants']

    # Plot the data
    plt.figure(figsize=(10, 6))

    # Apply custom colors based on the difference value
    colors = ['r' if x < 0 else '#F8C471' for x in ksa_diff_df['Difference']]

    bars = plt.bar(ksa_diff_df['Difference'], ksa_diff_df['Number of Participants'], color=colors)
    plt.xlabel('Difference', fontweight='bold')
    plt.ylabel('Number of Participants', fontweight='bold')
    plt.grid(False)

    # Replace x-axis labels with custom labels indicating level change
    custom_labels = [f"{int(level)} level increase" if level != 0 else "No change" for level in ksa_diff_df['Difference']]
    plt.xticks(ksa_diff_df['Difference'], custom_labels)

    # Add title
    plt.title('Distribution of Knowledge/Skills/Abilities Level Increase (AFTER - BEFORE) of the Course', fontweight='bold')

    # Annotate each bar with its count
    for bar, count in zip(bars, ksa_diff_df['Number of Participants']):
        plt.text(bar.get_x() + bar.get_width() / 2, count, f'{int(count)}', ha='center', va='bottom', fontsize=10)

## Generate a diagram for individual KSA improvement
def plot_individual_ksa_improvement(filtered_eval_info, eval_col_before, eval_col_after, eval_value, na_values=None):
    # Default NA values if none provided
    if na_values is None:
        na_values = ['nan', 'na', 'not applicable', '']

    filtered_eval_info = filtered_eval_info.copy()

    # Convert columns to string, lowercase, and replace specified values with np.nan
    filtered_eval_info[eval_col_before] = filtered_eval_info[eval_col_before].astype(str).str.lower().replace(na_values, np.nan)
    filtered_eval_info[eval_col_after] = filtered_eval_info[eval_col_after].astype(str).str.lower().replace(na_values, np.nan)

    # Ensure all entries are strings before mapping
    filtered_eval_info[eval_col_before] = filtered_eval_info[eval_col_before].astype(str)
    filtered_eval_info[eval_col_after] = filtered_eval_info[eval_col_after].astype(str)

    # Map the values to numeric labels
    value_to_label = {'none': '1', 'little': '2', 'basic': '3', 'intermediate': '4', 'advanced': '5'}
    
    # Use .loc to ensure changes are made in place on the DataFrame
    filtered_eval_info.loc[:, eval_col_before] = pd.to_numeric(filtered_eval_info[eval_col_before].map(value_to_label), errors='coerce')
    filtered_eval_info.loc[:, eval_col_after] = pd.to_numeric(filtered_eval_info[eval_col_after].map(value_to_label), errors='coerce')

    # Drop rows with NaNs in the mapped numeric values
    filtered_eval_info.dropna(subset=[eval_col_before, eval_col_after], inplace=True)

    # Filter the data based on the specified eval_value
    data_filtered = filtered_eval_info[filtered_eval_info[eval_col_before] == eval_value]

    if data_filtered.empty:
        print(f"No data available for Before Level = {eval_value}")
        return None

    # Define the number of columns in the grid
    num_columns = 6

    # Calculate the number of rows needed
    num_rows = len(data_filtered) // num_columns + (len(data_filtered) % num_columns > 0)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, max(2.5, num_rows * 2.5)))

    if num_rows * num_columns > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for index, row in enumerate(data_filtered.itertuples()):
        ax = axes[index]
        ax.bar('Before', getattr(row, eval_col_before), color='#9dcaea', alpha=0.8)
        ax.bar('After', getattr(row, eval_col_after), color='#2980B9')
        ax.plot(['Before', 'After'], [getattr(row, eval_col_before), getattr(row, eval_col_after)], 'ro-', linewidth=2)
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(False)

    for i in range(index + 1, num_rows * num_columns):
        axes[i].axis('off')

    plt.tight_layout()



