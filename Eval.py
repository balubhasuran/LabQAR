#Evaluation Script
import pandas as pd
import re
import random
import json

# Load the Excel file
file_path = 'file_path'
df = pd.read_excel(file_path, sheet_name=None)

# Extract the relevant sheet
evaluation_df = df['Sheet2']

# Rename the columns to simplify access
evaluation_df.columns = [
    'ID', 'Lab_test', 'Answer', 'GPT-4', 'GPT-3.5', 'GPT-4_Prediction', 'Reason_GPT-4_Prediction',
    'GPT-3.5_Prediction', 'Reason_GPT-3.5_Prediction', 'Answer_Set2', 'GPT-4_Set2', 'GPT-3.5_Set2',
    'GPT-4_Prediction_Set2', 'Reason_GPT-4_Prediction_Set2', 'GPT-3.5_Prediction_Set2', 'Reason_GPT-3.5_Prediction_Set2'
]

# Define function to evaluate the predictions with lenient matching
def lenient_evaluate_prediction(answer, prediction):
    if pd.isna(prediction):
        return "N/A"

    # Check if the answer and prediction are in range format (e.g., "0.5-1.2")
    if '-' in str(answer) and '-' in str(prediction):
        answer_range = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", answer)]
        prediction_range = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", prediction)]

        if len(answer_range) == 2 and len(prediction_range) == 2:
            if prediction_range[0] <= answer_range[0] and prediction_range[1] >= answer_range[1]:
                return "Correct"
            elif prediction_range[0] <= answer_range[1] and prediction_range[1] >= answer_range[0]:
                return "Correct"
            else:
                return "Incorrect"

    # Check if the answer is a single value within a prediction range (e.g., "21.7" in "0-21.7")
    if '-' in str(prediction):
        prediction_range = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", prediction)]
        if len(prediction_range) == 2 and prediction_range[0] <= float(answer) <= prediction_range[1]:
            return "Correct"

    return "Correct" if answer == prediction else "Incorrect"

def reasoning(answer, prediction):
    if pd.isna(prediction):
        return "Prediction not available"
    return "Prediction matches the answer" if answer == prediction else "Prediction does not match the answer"

# Update the prediction columns with lenient evaluation
evaluation_df['GPT-4_Prediction'] = evaluation_df.apply(lambda row: lenient_evaluate_prediction(row['Answer'], row['GPT-4']), axis=1)
evaluation_df['Reason_GPT-4_Prediction'] = evaluation_df.apply(lambda row: reasoning(row['Answer'], row['GPT-4']), axis=1)
evaluation_df['GPT-3.5_Prediction'] = evaluation_df.apply(lambda row: lenient_evaluate_prediction(row['Answer'], row['GPT-3.5']), axis=1)
evaluation_df['Reason_GPT-3.5_Prediction'] = evaluation_df.apply(lambda row: reasoning(row['Answer'], row['GPT-3.5']), axis=1)

evaluation_df['GPT-4_Prediction_Set2'] = evaluation_df.apply(lambda row: lenient_evaluate_prediction(row['Answer_Set2'], row['GPT-4_Set2']), axis=1)
evaluation_df['Reason_GPT-4_Prediction_Set2'] = evaluation_df.apply(lambda row: reasoning(row['Answer_Set2'], row['GPT-4_Set2']), axis=1)
evaluation_df['GPT-3.5_Prediction_Set2'] = evaluation_df.apply(lambda row: lenient_evaluate_prediction(row['Answer_Set2'], row['GPT-3.5_Set2']), axis=1)
evaluation_df['Reason_GPT-3.5_Prediction_Set2'] = evaluation_df.apply(lambda row: reasoning(row['Answer_Set2'], row['GPT-3.5_Set2']), axis=1)

# Define function to map prediction results to 0 and 1
def map_results_to_binary(result):
    if result == "Correct":
        return 1
    elif result == "Incorrect":
        return 0
    else:
        return None

# Apply the mapping function to the prediction columns
evaluation_df['GPT-4_Prediction_Binary'] = evaluation_df['GPT-4_Prediction'].map(map_results_to_binary)
evaluation_df['GPT-3.5_Prediction_Binary'] = evaluation_df['GPT-3.5_Prediction'].map(map_results_to_binary)
evaluation_df['GPT-4_Prediction_Set2_Binary'] = evaluation_df['GPT-4_Prediction_Set2'].map(map_results_to_binary)
evaluation_df['GPT-3.5_Prediction_Set2_Binary'] = evaluation_df['GPT-3.5_Prediction_Set2'].map(map_results_to_binary)

# Calculate the accuracy
gpt_4_accuracy_lenient = evaluation_df['GPT-4_Prediction_Binary'].mean()
gpt_3_5_accuracy_lenient = evaluation_df['GPT-3.5_Prediction_Binary'].mean()
gpt_4_accuracy_set2_lenient = evaluation_df['GPT-4_Prediction_Set2_Binary'].mean()
gpt_3_5_accuracy_set2_lenient = evaluation_df['GPT-3.5_Prediction_Set2_Binary'].mean()

# Save the results to a new Excel file
output_file_path_lenient = 'file_path'
evaluation_df.to_excel(output_file_path_lenient, index=False)

(gpt_4_accuracy_lenient, gpt_3_5_accuracy_lenient, gpt_4_accuracy_set2_lenient, gpt_3_5_accuracy_set2_lenient, output_file_path_lenient)

#Code 2
import pandas as pd

# Load the Excel file provided by the user
file_path = 'file_path'
df = pd.read_excel(file_path, sheet_name='Sheet2')

# Function to handle different formats of values
def process_range(value):
    if isinstance(value, str):
        value = value.strip()
        if '-' in value:
            try:
                start, end = map(float, value.split('-'))
                return (start + end) / 2
            except ValueError:
                print(f"Could not convert range value: {value}")
                return None
        elif '<' in value:
            try:
                return float(value.replace('<', '').strip()) / 2  # Take half of the value if it's a '<' case
            except ValueError:
                print(f"Could not convert less-than value: {value}")
                return None
        elif '>' in value:
            try:
                return float(value.replace('>', '').strip()) * 2  # Double the value if it's a '>' case
            except ValueError:
                print(f"Could not convert greater-than value: {value}")
                return None
    try:
        return float(value)
    except ValueError:
        print(f"Could not convert value: {value}")
        return None

# Apply the processing function to Answer, GPT-4, and GPT-3.5 columns
df['Answer'] = df['Answer'].apply(process_range)
df['GPT-4'] = df['GPT-4'].apply(process_range)
df['GPT-3.5'] = df['GPT-3.5'].apply(process_range)

# Recalculate strict match for GPT-4 and GPT-3.5
df['GPT-4 Prediction_Strict Match'] = (df['Answer'] == df['GPT-4']).astype(int)
df['GPT-3.5 Prediction_Strict Match'] = (df['Answer'] == df['GPT-3.5']).astype(int)

# Recalculate lenient 10% match for GPT-4 and GPT-3.5
df['GPT-4 Prediction_Linient10 Match'] = ((df['GPT-4'] <= df['Answer'] * 0.10) & (df['GPT-4'] >= df['Answer'] * 0.10)).astype(int)
df['GPT-3.5 Prediction_Linient10 Match'] = ((df['GPT-3.5'] <= df['Answer'] * 0.10) & (df['GPT-3.5'] >= df['Answer'] * 0.10)).astype(int)

# Recalculate lenient 20% match for GPT-4 and GPT-3.5
df['GPT-4 Prediction_Linient20 Match'] = ((df['GPT-4'] <= df['Answer'] * 0.20) & (df['GPT-4'] >= df['Answer'] * 0.20)).astype(int)
df['GPT-3.5 Prediction_Linient20 Match'] = ((df['GPT-3.5'] <= df['Answer'] * 0.20) & (df['GPT-3.5'] >= df['Answer'] * 0.20)).astype(int)

# Recalculate accuracy for each prediction type
accuracy_strict_gpt4 = df['GPT-4 Prediction_Strict Match'].mean()
accuracy_strict_gpt35 = df['GPT-3.5 Prediction_Strict Match'].mean()
accuracy_lenient10_gpt4 = df['GPT-4 Prediction_Linient10 Match'].mean()
accuracy_lenient10_gpt35 = df['GPT-3.5 Prediction_Linient10 Match'].mean()
accuracy_lenient20_gpt4 = df['GPT-4 Prediction_Linient20 Match'].mean()
accuracy_lenient20_gpt35 = df['GPT-3.5 Prediction_Linient20 Match'].mean()

accuracies = {
    'GPT-4 Prediction_Strict Match': accuracy_strict_gpt4,
    'GPT-3.5 Prediction_Strict Match': accuracy_strict_gpt35,
    'GPT-4 Prediction_Linient10 Match': accuracy_lenient10_gpt4,
    'GPT-3.5 Prediction_Linient10 Match': accuracy_lenient10_gpt35,
    'GPT-4 Prediction_Linient20 Match': accuracy_lenient20_gpt4,
    'GPT-3.5 Prediction_Linient20 Match': accuracy_lenient20_gpt35,
}

# Print the accuracies
print("Accuracies:")
for key, value in accuracies.items():
    print(f"{key}: {value:.2%}")

df.head(10)

import pandas as pd

# Data
file_path = 'file_path'
df = pd.read_excel(file_path, sheet_name='Sheet2')

def parse_range(range_str):
    """Parses a range string or single value and returns the lower and upper bounds as floats."""
    if '-' in range_str:
        lower, upper = map(float, range_str.split('-'))
    else:
        lower = upper = float(range_str)
    return lower, upper

def check_bounds(answer_range, test_range, tolerance):
    """Checks if the test range is within the tolerance bounds of the answer range."""
    ans_lower, ans_upper = parse_range(answer_range)
    test_lower, test_upper = parse_range(test_range)

    lower_bound = ans_lower * (1 - tolerance)
    upper_bound = ans_upper * (1 + tolerance)

    return lower_bound <= test_lower <= upper_bound and lower_bound <= test_upper <= upper_bound

# Check for matches
results = []
tolerances = [0, 0.10, 0.20]

for index, row in df.iterrows():
    answer = row['Answer']
    gpt_4 = row['GPT-4']
    gpt_35 = row['GPT-3.5']

    result = {
        'Lab test': row['Lab test'],
        'Answer':row['Answer'],
        'GPT_4' :row['GPT-4'],
        'GPT_3.5' :row['GPT-3.5'],
        'Exact match GPT-4': check_bounds(answer, gpt_4, 0),
        'Exact match GPT-3.5': check_bounds(answer, gpt_35, 0),
        'Within 10% GPT-4': check_bounds(answer, gpt_4, 0.10),
        'Within 10% GPT-3.5': check_bounds(answer, gpt_35, 0.10),
        'Within 20% GPT-4': check_bounds(answer, gpt_4, 0.20),
        'Within 20% GPT-3.5': check_bounds(answer, gpt_35, 0.20),
    }

    results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Replace True and False with 1 and 0
results_df = results_df.replace({True: 1, False: 0})

results_df

results_df.to_csv('file_path')

import pandas as pd

# Data
file_path = 'file_path'
df = pd.read_excel(file_path, sheet_name='Sheet2')

# Define the function to parse range strings or single values
def parse_range(range_str):
    """Parses a range string or single value and returns the lower and upper bounds as floats."""
    if isinstance(range_str, (int, float)):
        lower = upper = float(range_str)
    elif not range_str or range_str.strip() == '':
        # Return None for invalid or empty range strings
        return None, None
    elif '-' in range_str:
        try:
            parts = [p for p in range_str.split('-') if p.strip()]  # Remove empty parts
            lower, upper = float(parts[0]), float(parts[-1])
        except (ValueError, IndexError):
            # Handle cases where the value cannot be converted to float
            return None, None
    else:
        try:
            lower = upper = float(range_str)
        except ValueError:
            return None, None
    return lower, upper

# Define the function to check bounds with a given tolerance
def check_bounds(answer_range, test_range, tolerance):
    """Checks if the test range is within the tolerance bounds of the answer range."""
    ans_lower, ans_upper = parse_range(answer_range)
    test_lower, test_upper = parse_range(test_range)

    if ans_lower is None or test_lower is None:
        return False

    lower_bound = ans_lower * (1 - tolerance)
    upper_bound = ans_upper * (1 + tolerance)

    return lower_bound <= test_lower <= upper_bound and lower_bound <= test_upper <= upper_bound

# Analyze the data for exact matches and within tolerance matches
results = []

for index, row in df.iterrows():
    answer = row['Answer']
    gpt_4 = row['GPT-4']
    gpt_35 = row['GPT-3.5']
    llama_3 = row['LLaMA3.1']  # Column name for LLaMA 3 is 'LLaMA3.1'
    gpt_4o = row['GPT-4o']  # Include GPT-4 RAG

    result = {
        'Lab test': row['Lab test'],
        'Answer': row['Answer'],
        'GPT_4': row['GPT-4'],
        'GPT_3.5': row['GPT-3.5'],
        'LLaMA 3': row['LLaMA3.1'],  # Add LLaMA 3 to the result
        'GPT-4o': row['GPT-4o'],  # Add GPT-4 RAG to the result
        'Exact match GPT-4': check_bounds(answer, gpt_4, 0),
        'Exact match GPT-3.5': check_bounds(answer, gpt_35, 0),
        'Exact match LLaMA 3': check_bounds(answer, llama_3, 0),  # Exact match for LLaMA 3
        'Exact match GPT-4o': check_bounds(answer, gpt_4o, 0),  # Exact match for GPT-4 RAG
        'Within 10% GPT-4': check_bounds(answer, gpt_4, 0.10),
        'Within 10% GPT-3.5': check_bounds(answer, gpt_35, 0.10),
        'Within 10% LLaMA 3': check_bounds(answer, llama_3, 0.10),  # 10% match for LLaMA 3
        'Within 10% GPT-4o': check_bounds(answer, gpt_4o, 0.10),  # 10% match for GPT-4 RAG
        'Within 20% GPT-4': check_bounds(answer, gpt_4, 0.20),
        'Within 20% GPT-3.5': check_bounds(answer, gpt_35, 0.20),
        'Within 20% LLaMA 3': check_bounds(answer, llama_3, 0.20),  # 20% match for LLaMA 3
        'Within 20% GPT-4o': check_bounds(answer, gpt_4o, 0.20),  # 20% match for GPT-4 RAG
    }

    results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Replace True and False with 1 and 0 for easier analysis
results_df = results_df.replace({True: 1, False: 0})
results_df.to_csv('D:\\res.csv')

# Calculate the sums
n = 550
results_summary = {
    'LLM': ['GPT-3.5', 'GPT-4', 'LLaMA 3', 'GPT-4 RAG'],
    'Exact Match': [
        f"{results_df['Exact match GPT-3.5'].sum()/n:.3f}({results_df['Exact match GPT-3.5'].sum()}/{n})",
        f"{results_df['Exact match GPT-4'].sum()/n:.3f}({results_df['Exact match GPT-4'].sum()}/{n})",
        f"{results_df['Exact match LLaMA 3'].sum()/n:.3f}({results_df['Exact match LLaMA 3'].sum()}/{n})",
        f"{results_df['Exact match GPT-4o'].sum()/n:.3f}({results_df['Exact match GPT-4o'].sum()}/{n})"
    ],
    '10% Lower/Upper': [
        f"{results_df['Within 10% GPT-3.5'].sum()/n:.3f}({results_df['Within 10% GPT-3.5'].sum()}/{n})",
        f"{results_df['Within 10% GPT-4'].sum()/n:.3f}({results_df['Within 10% GPT-4'].sum()}/{n})",
        f"{results_df['Within 10% LLaMA 3'].sum()/n:.3f}({results_df['Within 10% LLaMA 3'].sum()}/{n})",
        f"{results_df['Within 10% GPT-4o'].sum()/n:.3f}({results_df['Within 10% GPT-4o'].sum()}/{n})"
    ],
    '20% Lower/Upper': [
        f"{results_df['Within 20% GPT-3.5'].sum()/n:.3f}({results_df['Within 20% GPT-3.5'].sum()}/{n})",
        f"{results_df['Within 20% GPT-4'].sum()/n:.3f}({results_df['Within 20% GPT-4'].sum()}/{n})",
        f"{results_df['Within 20% LLaMA 3'].sum()/n:.3f}({results_df['Within 20% LLaMA 3'].sum()}/{n})",
        f"{results_df['Within 20% GPT-4o'].sum()/n:.3f}({results_df['Within 20% GPT-4o'].sum()}/{n})"
    ]
}

results_summary_df = pd.DataFrame(results_summary)
results_summary_df

# Load the data
file_path = 'file_path'
df = pd.read_excel(file_path)

# Perform the comparison
df['GPT-4_Prediction_Set2_Binary'] = (df['Answer_Set2'] == df['GPT-4_Set2']).astype(int)
df['GPT-3.5_Prediction_Set2_Binary'] = (df['Answer_Set2'] == df['GPT-3.5_Set2']).astype(int)
df['LLaMA3_Prediction_Set2_Binary'] = (df['Answer_Set2'] == df['LLaMA3_Set2']).astype(int)
df['GPT-4_RAG_Prediction_Set2_Binary'] = (df['Answer_Set2'] == df['GPT-4_RAG_Set2']).astype(int)
# Display the updated DataFrame
df.to_excel('file_path')
df.head()

n = 550
accuracy_results = {
    'LLM': ['GPT-4', 'GPT-3.5', 'LLaMA 3', 'GPT-4 RAG'],
    'Accuracy': [
        f"{df['GPT-4_Prediction_Set2_Binary'].sum()/n:.3f}({df['GPT-4_Prediction_Set2_Binary'].sum()}/{n})",
        f"{df['GPT-3.5_Prediction_Set2_Binary'].sum()/n:.3f}({df['GPT-3.5_Prediction_Set2_Binary'].sum()}/{n})",
        f"{df['LLaMA3_Prediction_Set2_Binary'].sum()/n:.3f}({df['LLaMA3_Prediction_Set2_Binary'].sum()}/{n})",
         f"{df['GPT-4_RAG_Prediction_Set2_Binary'].sum()/n:.3f}({df['GPT-4_RAG_Prediction_Set2_Binary'].sum()}/{n})"
    ]
}

accuracy_df = pd.DataFrame(accuracy_results)
accuracy_df

import json

# Load the JSON data
with open('file_path', 'r') as json_file:
    json_data = json.load(json_file)

# Load the processed predictions
with open('file_path', 'r') as processed_file:
    predictions = processed_file.readlines()

# Function to map predictions to choices based on the JSON file's Choices field
def map_prediction_to_choice(prediction, choices):
    prediction = prediction.strip().lower()
    choices_dict = {choice.split(':')[1].strip().lower(): choice.split(':')[0].strip() for choice in choices.split('\n') if choice}

    if 'normal' in prediction:
        return choices_dict.get('normal', 'D')
    elif 'low' in prediction:
        return choices_dict.get('low', 'D')
    elif 'high' in prediction:
        return choices_dict.get('high', 'D')
    else:
        return 'D'


# Iterate over JSON data and map predictions, print mapped choices
for i, item in enumerate(json_data):
    if i < len(predictions):
        prediction = predictions[i]
        choices = item['Choices']
        mapped_choice = map_prediction_to_choice(prediction, choices)
        print(mapped_choice)
# Iterate over JSON data and map predictions
# for i, item in enumerate(json_data):
#     if i < len(predictions):
#         prediction = predictions[i]
#         choices = item['Choices']
#         mapped_choice = map_prediction_to_choice(prediction, choices)
#         print(f"ID: {item['ID']}, Prediction: {prediction.strip()}, Mapped Choice: {mapped_choice}")

#         # Validate against the answer
#         if mapped_choice == item['Answer']:
#             print(f"Result: Correct (Answer: {item['Answer']})\n")
#         else:
#             print(f"Result: Incorrect (Answer: {item['Answer']})\n")

#GatorTronGPT
import pandas as pd
import json

# Load the JSONL file
file_path = 'file_path'

# Read the JSONL file
data = []
with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract the prediction and answer
df['Prediction'] = df['pred'].str.strip()
df['Answer'] = df['label'].str.strip()

# Compare the prediction and answer
df['Result'] = (df['Prediction'] == df['Answer']).astype(int)

df.head(20)

df['Result'].value_counts()

# Calculate the accuracy
accuracy = df['Result'].mean()
accuracy

import matplotlib.pyplot as plt

# Plot the distribution of predictions
plt.figure(figsize=(10, 6))
df['Prediction'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Predictions')
plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Plot the correct vs incorrect predictions
plt.figure(figsize=(10, 6))
df['Result'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Correct vs Incorrect Predictions')
plt.xlabel('Result')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['Incorrect', 'Correct'], rotation=0)
plt.show()

import pandas as pd
import json

# Load the JSONL file
file_path = 'file_pathl'

# Read the JSONL file
data = []
with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the parsing and checking functions
def parse_range_safe(range_str):
    """Parses a range string or single value and returns the lower and upper bounds as floats. Handles errors safely."""
    try:
        if '-' in range_str:
            lower, upper = map(float, range_str.split('-'))
        else:
            lower = upper = float(range_str)
        return lower, upper
    except ValueError:
        return None, None

def check_bounds_safe(answer_range, test_range, tolerance):
    """Checks if the test range is within the tolerance bounds of the answer range, handling errors safely."""
    ans_lower, ans_upper = parse_range_safe(answer_range)
    test_lower, test_upper = parse_range_safe(test_range)

    if None in (ans_lower, ans_upper, test_lower, test_upper):
        return False

    lower_bound = ans_lower * (1 - tolerance)
    upper_bound = ans_upper * (1 + tolerance)

    return lower_bound <= test_lower <= upper_bound and lower_bound <= test_upper <= upper_bound

# Extract the prediction and answer
df['Prediction'] = df['pred'].str.strip()
df['Answer'] = df['label'].str.strip()

# Check for matches with error handling
results = []

for index, row in df.iterrows():
    answer = row['Answer']
    prediction = row['Prediction']

    result = {
        'Prediction': row['Prediction'],
        'Answer': row['Answer'],
        'Exact match': check_bounds_safe(answer, prediction, 0),
        'Within 10%': check_bounds_safe(answer, prediction, 0.10),
        'Within 20%': check_bounds_safe(answer, prediction, 0.20),
    }

    results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

results_df.head(10)

results_df['Exact match'].value_counts()

results_df['Within 10%'].value_counts()

results_df['Within 20%'].value_counts()

# Update the results to use 1 and 0 instead of True and False
results_df = results_df.replace({True: 1, False: 0})

# Calculate the accuracy for each match type
exact_match_accuracy = results_df['Exact match'].mean()
within_10_accuracy = results_df['Within 10%'].mean()
within_20_accuracy = results_df['Within 20%'].mean()

accuracies = {
    'Exact match accuracy': exact_match_accuracy,
    'Within 10% accuracy': within_10_accuracy,
    'Within 20% accuracy': within_20_accuracy
}

accuracies

# Plot the distribution of predictions
plt.figure(figsize=(10, 6))
df['Prediction'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Predictions')
plt.xlabel('Prediction')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Visualize correct vs incorrect predictions for exact match, within 10%, and within 20%
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Exact match
results_df['Exact match'].value_counts().plot(kind='bar', color=['salmon', 'skyblue'], ax=axes[0])
axes[0].set_title('Exact Match')
axes[0].set_xlabel('Result')
axes[0].set_ylabel('Frequency')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Incorrect', 'Correct'])

# Within 10%
results_df['Within 10%'].value_counts().plot(kind='bar', color=['salmon', 'skyblue'], ax=axes[1])
axes[1].set_title('Within 10%')
axes[1].set_xlabel('Result')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Incorrect', 'Correct'])

# Within 20%
results_df['Within 20%'].value_counts().plot(kind='bar', color=['salmon', 'skyblue'], ax=axes[2])
axes[2].set_title('Within 20%')
axes[2].set_xlabel('Result')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Incorrect', 'Correct'])

plt.tight_layout()
plt.show()

# Define the function to categorize predictions into different ranges
def categorize_prediction(range_str):
    lower, upper = parse_range_safe(range_str)
    if lower is None or upper is None:
        return 'Invalid'

    if upper <= 10:
        return '0-10'
    elif upper <= 20:
        return '11-20'
    elif upper <= 50:
        return '21-50'
    elif upper <= 100:
        return '51-100'
    else:
        return '101+'

# Apply the function to categorize predictions
df['Prediction Range'] = df['Prediction'].apply(categorize_prediction)

# Calculate the distribution of predictions by different ranges
range_distribution = df['Prediction Range'].value_counts()

range_distribution

# Apply the function to categorize answers
df['Answer Range'] = df['Answer'].apply(categorize_prediction)

# Calculate the distribution of answers by different ranges
answer_range_distribution = df['Answer Range'].value_counts()

# Calculate percentages for each answer range
answer_range_distribution_percent = (answer_range_distribution / answer_range_distribution.sum()) * 100

# Plot the distribution of answer ranges by percentages
plt.figure(figsize=(10, 6))
answer_range_distribution_percent.plot(kind='bar', color='salmon')
plt.title('Distribution of Answer Ranges by Percentages')
plt.xlabel('Answer Range')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.show()

# Create a crosstab to compare prediction ranges vs answer ranges
prediction_vs_answer = pd.crosstab(df['Prediction Range'], df['Answer Range'])

# Plot the crosstab as a heatmap
plt.figure(figsize=(12, 8))
plt.imshow(prediction_vs_answer, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Frequency')
plt.title('Prediction Ranges vs Answer Ranges')
plt.xlabel('Answer Range')
plt.ylabel('Prediction Range')
plt.xticks(ticks=range(len(prediction_vs_answer.columns)), labels=prediction_vs_answer.columns, rotation=45)
plt.yticks(ticks=range(len(prediction_vs_answer.index)), labels=prediction_vs_answer.index)
plt.show()
