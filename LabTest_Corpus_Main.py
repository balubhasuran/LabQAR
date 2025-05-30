import pandas as pd
import random
import json


# Load the Excel file
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Labs Range_References\\Labs_LOINC.xlsx'
labs_df = pd.read_excel(file_path)

# Display the first few rows to understand its structure
labs_df.head()



# Adjusting the generate_options function to directly include it in the loop
def generate_options_with_context(correct_range):
    try:
        if '–' in correct_range:  # For ranges
            low, high = map(float, correct_range.split('–'))
            distractors = [
                f"{round(low*0.9, 1)}–{round(high*0.9, 1)}",
                f"{round(low*1.1, 1)}–{round(high*1.1, 1)}",
                f"{low}–{round(high*1.2, 1)}"
            ]
        elif correct_range.startswith('<'):  # For upper limits
            value = float(correct_range[1:])
            distractors = [
                f"<{round(value*0.8, 1)}",
                f"<{round(value*1.2, 1)}",
                f"<{round(value*1.5, 1)}"
            ]
        else:  # For exact values
            value = float(correct_range)
            distractors = [
                str(round(value*0.8, 1)),
                str(round(value*1.2, 1)),
                str(round(value*1.5, 1))
            ]
    except ValueError:
        distractors = [f"Not {correct_range}", f"Above {correct_range}", f"Below {correct_range}"]

    options = [correct_range] + distractors
    random.shuffle(options)
    return options

# Initialize new questions and answers with additional details
new_questions = []
new_correct_answers = []

for index, row in labs_df.iterrows():
    options = generate_options_with_context(row['SI Reference Interval'])
    correct_option_index = options.index(row['SI Reference Interval'])  # Find the index of the correct answer

    # Construct the detailed question text with additional column values
    gender_specific = f" for ‘{row['Gender-specific']}’" if pd.notna(row['Gender-specific']) else ""
    age_specific = f" in their ‘{row['Age group-specific']}’" if pd.notna(row['Age group-specific']) else ""
    women_condition = f" in their ‘{row['Women-related condition']}’" if pd.notna(row['Women-related condition']) else ""
    question_text = f"Q{index+1}: For the lab test '{row['Lab Test']}' measuring in '{row['SI Units']}' in Specimen ‘{row['Specimen']}’{gender_specific}{age_specific}{women_condition}. what is the correct SI reference range?"
    options_text = "\n".join([f"{chr(65+i)}: {option}" for i, option in enumerate(options)])
    correct_answer_text = f"Correct answer for Q{index+1} ('{row['Lab Test']}'): Option {chr(65+correct_option_index)}"

    new_questions.append(f"{question_text}\nOptions:\n{options_text}")
    new_correct_answers.append(correct_answer_text)

# Combine new questions and their correct answers into a JSON structure for the example
example_question_json = {
    "questions": [
        {"question": new_questions[0], "correct_answer": new_correct_answers[0]}
    ]
}

# Example output for the first question
example_question_json

# Now generating the full JSON structure for all questions and answers with the updated format
full_questions_json = {
    "questions": [
        {"question": new_questions[i], "correct_answer": new_correct_answers[i]}
        for i in range(len(new_questions))
    ]
}

# Define the new JSON file path
full_json_file_path = 'C://D//full_lab_test_questions.json'

# Write the full JSON data to a file
with open(full_json_file_path, 'w') as json_file:
    json.dump(full_questions_json, json_file, indent=4)

full_json_file_path

#select 10 most frequent labs
ls=['Albumin','Creatine','Hemoglobin A1C','Hemoglobin','Calcium','Bilirubin','Blood urea nitrogen (BUN)','Cholesterol','High-density lipoprotein','Low-density lipoprotein','Triglycerides','Thyroid-stimulating hormone (TSH)','C-reactive protein','Vitamin D','Ferritin']
ls

df_10=labs_df[labs_df['Lab Test'].isin(ls)]

df_10

df_10.to_excel('C://D//10_Labs.xlsx')

import pandas as pd

# Load the Excel sheet
file_path = 'C://D//10_Labs.xlsx'
labs_df = pd.read_excel(file_path)

# Check the first few rows to understand the data structure
labs_df.head()

from random import choice, shuffle, randint
import numpy as np

# Function to scale the reference range
def scale_reference_range(ref_range, factor):
    try:
        low, high = ref_range.split('–')
        low, high = float(low) * factor, float(high) * factor
        return f"{low:.1f}–{high:.1f}"
    except ValueError:
        # Handle cases like "<0.2" or ">100"
        if '<' in ref_range:
            value = float(ref_range.split('<')[1]) * factor
            return f"<{value:.1f}"
        elif '>' in ref_range:
            value = float(ref_range.split('>')[1]) * factor
            return f">{value:.1f}"
        else:
            return ref_range

# Function to convert units (here we simulate it by dividing by 1,000,000 for nmol/L to mol/L)
def convert_units(ref_range):
    try:
        low, high = ref_range.split('–')
        low, high = float(low) / 1000000, float(high) / 1000000
        return f"{low:.1e}–{high:.1e}"
    except ValueError:
        # Handle cases like "<0.2" or ">100"
        if '<' in ref_range:
            value = float(ref_range.split('<')[1]) / 1000000
            return f"<{value:.1e}"
        elif '>' in ref_range:
            value = float(ref_range.split('>')[1]) / 1000000
            return f">{value:.1e}"
        else:
            return ref_range

questions = []

for index, row in labs_df.iterrows():
    correct_answer = row['SI Reference Interval']
    specimen = row['Specimen']
    gender_specific = row['Gender-specific'] if pd.notnull(row['Gender-specific']) else "any gender"
    age_group_specific = row['Age group-specific'] if pd.notnull(row['Age group-specific']) else "any age group"
    lab_test = row['Lab Test']

    # Incorrect options
    options = [correct_answer]

    # Option from another lab test's correct range
    other_test_correct_range = choice(labs_df[labs_df.index != index]['SI Reference Interval'].values)
    options.append(other_test_correct_range)

    # Option by scaling the correct SI reference range
    scaled_range = scale_reference_range(correct_answer, 10)
    options.append(scaled_range)

    # Option by converting units
    converted_range = convert_units(correct_answer)
    options.append(converted_range)

    shuffle(options)
    correct_option = 'ABCD'[options.index(correct_answer)]

    question_text = f"For the lab test '{lab_test}' measuring in '{row['SI Units']}' in Specimen '{specimen}' for '{gender_specific}', what is the correct SI reference range?"

    questions.append({
        "ID": index + 1,
        "Question": question_text,
        "Choices": "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)]),
        "Answer": correct_option
    })

questions[:15]  # Show first 3 questions as a sample

# Adjust the process based on the new instructions for conversion (e.g., nmol/L to ng/ml)

# Helper function to convert SI reference range for specific conversion scenario (e.g., nmol/L to ng/ml)
# For simplicity and general applicability, we'll simulate this conversion process
def simulate_conversion(ref_range):
    # Attempt to split the range and apply a conversion factor
    try:
        low, high = ref_range.split('–')
        low, high = float(low) * 1.5, float(high) * 1.5  # Simulated conversion factor
        return f"{low:.1f}–{high:.1f}"
    except ValueError:
        # Handle cases like "<0.2" or ">100" with a simulated conversion
        if '<' in ref_range:
            value = float(ref_range.split('<')[1]) * 1.5
            return f"<{value:.1f}"
        elif '>' in ref_range:
            value = float(ref_range.split('>')[1]) * 1.5
            return f">{value:.1f}"
        else:
            return ref_range

# Initialize questions list
questions = []

for index, row in labs_df.iterrows():
    correct_answer = row['SI Reference Interval']
    specimen = row['Specimen']
    gender_specific = row['Gender-specific'] if pd.notnull(row['Gender-specific']) else "any gender"
    age_group_specific = row['Age group-specific'] if pd.notnull(row['Age group-specific']) else "any age group"
    lab_test = row['Lab Test']

    # Generate distinct incorrect options
    options = [correct_answer]

    # Keep trying to add unique options until we have 4 distinct ones
    while len(set(options)) < 4:
        # Option from another lab test's correct range
        other_test_correct_range = choice(labs_df['SI Reference Interval'].values)
        if other_test_correct_range not in options:
            options.append(other_test_correct_range)

        # Option by scaling the correct SI reference range by multiplying with value 10
        scaled_range = scale_reference_range(correct_answer, 10)
        if scaled_range not in options:
            options.append(scaled_range)

        # Option by converting a value to a new measure (e.g., nmol/L to ng/ml)
        converted_range = simulate_conversion(correct_answer)
        if converted_range not in options:
            options.append(converted_range)

    # Ensure options list has unique values only
    unique_options = list(set(options))[:4]
    shuffle(unique_options)  # Shuffle to randomize position of the correct answer
    correct_option = 'ABCD'[unique_options.index(correct_answer)]

    question_text = f"For the lab test '{lab_test}' measuring in '{row['SI Units']}' in Specimen '{specimen}' for '{gender_specific}', what is the correct SI reference range?"

    questions.append({
        "ID": index + 1,
        "Question": question_text,
        "Choices": "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(unique_options)]),
        "Answer": correct_option
    })

# Return the corrected questions list
questions[:15]  # Show first 3 questions as a sample

questions

import json

# Since the initial attempt to directly execute the file content as Python code wasn't successful, let's parse it manually.
# Given the structured nature of the data, we'll read the file, parse it accordingly, and convert it to JSON format.

file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Corpus_15Labs.txt'

# Initialize an empty list to hold the extracted data
data = []

# Open the file and read the contents
with open(file_path, 'r') as file:
    content = file.read()
    # The content seems to be a string representation of a list of dictionaries
    data = eval(content)

# Convert the list of dictionaries to JSON format
json_data = json.dumps(data, indent=4)

# Optionally, you can save the JSON data to a new file
json_file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Corpus_15Labs.json'
with open(json_file_path, 'w') as json_file:
    json_file.write(json_data)

# Provide the path to the saved JSON file
json_file_path

#Normal low or high
import pandas as pd
import re

# Load the Excel file
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\10_Labs.xlsx'
df = pd.read_excel(file_path)


def generate_random_value_adjusted(reference_interval):
    """
    Adjusted to generate a random value more carefully, considering different formats of reference intervals.
    """
    try:
        if reference_interval.startswith("<"):
            max_val = float(reference_interval[1:])
            return random.uniform(max_val, max_val * 1.5), "High"
        elif reference_interval.startswith(">"):
            min_val = float(reference_interval[1:])
            return random.uniform(0, min_val * 0.5), "Low"
        else:
            min_val, max_val = map(float, re.findall(r"[\d.]+", reference_interval))
            choice = random.choice(["Normal", "Low", "High"])
            if choice == "Normal":
                value = random.uniform(min_val, max_val)
            elif choice == "Low":
                value = random.uniform(min_val * 0.5, min_val)
            else:  # "High"
                value = random.uniform(max_val, max_val * 1.5)
            return value, choice
    except ValueError:
        return 0, "Normal"

def shuffle_choices(correct_classification):
    choices = [("A", "Normal"), ("B", "Low"), ("C", "High")]
    # Extract conditions and shuffle them
    conditions = [choice[1] for choice in choices]
    random.shuffle(conditions)
    # Reassign shuffled conditions to the choice labels
    shuffled_choices = [(choices[i][0], conditions[i]) for i in range(len(choices))]

    # Find the letter corresponding to the correct answer after shuffling
    correct_answer = next(letter for letter, condition in shuffled_choices if condition == correct_classification)

    # Format the shuffled choices into a string for display
    choices_str = "\n".join([f"{letter}: {condition}" for letter, condition in shuffled_choices])
    return choices_str, correct_answer

questions_adjusted = []
for index, row in df.iterrows():
    random_value, classification = generate_random_value_adjusted(row['SI Reference Interval'])

    choices_str, correct_answer = shuffle_choices(classification)

    question_adjusted = {
        "ID": index + 1,
        "Question": f"Q{index + 1}: For the lab test '{row['Lab Test']}' measuring in '{row['SI Units']}' "
                    f"in Specimen ‘{row['Specimen']}’ for "
                    f"{'any gender' if pd.isna(row['Gender-specific']) else row['Gender-specific']} "
                    f"{'any age group' if pd.isna(row['Age group-specific']) else row['Age group-specific']} "
                    f"a value in 'SI reference range' is {random_value:.2f}. Is the lab test result is?",
        "Choices": choices_str,
        "Answer": correct_answer
    }

    questions_adjusted.append(question_adjusted)

# Save the adjusted questions as a JSON file
questions_json_adjusted = json.dumps({"questions": questions_adjusted}, indent=2)
questions_json_path_adjusted = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\10_lab_questions_normal_low_high1.json'
with open(questions_json_path_adjusted, 'w') as file:
    file.write(questions_json_adjusted)

questions_json_path_adjusted

question_adjusted

#Making option A as the correct answer
#Set 1 four options ('A','B','C','D')

import pandas as pd
import re

# Load the Excel file
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\10_Labs.xlsx'
df = pd.read_excel(file_path)
df

#Updation
from random import choice, randint
import re
import pandas as pd


# Load the Excel file
data = pd.read_excel('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\10_Labs.xlsx')

def convert_interval_to_scaled(value, scale):
    """Scales the numerical parts of the interval by a given scale factor."""
    parts = re.split('(\d+\.\d+|\d+)', value)
    scaled_parts = [str(float(part) * scale) if part.replace('.', '', 1).isdigit() else part for part in parts]
    return ''.join(scaled_parts)

def convert_to_different_measure(value, from_unit, to_unit):
    """Converts the numerical parts of the interval to a different measure."""
    parts = re.split('(\d+\.\d+|\d+)', value)
    conversion = {'μmol/L': {'mol/L': 1e-6}, 'mg/dL': {'g/L': 0.01}, 'mmol/L': {'mol/L': 1e-3}, 'g/L': {'mol/L': 1e-3}}
    if from_unit in conversion and to_unit in conversion[from_unit]:
        factor = conversion[from_unit][to_unit]
        converted_parts = [str(float(part) * factor) if part.replace('.', '', 1).isdigit() else part for part in parts]
        return ''.join(converted_parts)
    return value

def format_decimal_value(value):
    """Formats the numerical parts of the string to have only 3 decimal places."""
    parts = re.split('(\d+\.\d+|\d+)', value)
    formatted_parts = [f"{float(part):.3f}" if part.replace('.', '', 1).isdigit() else part for part in parts]
    return ''.join(formatted_parts)

def generate_questions_decimal_fix(data):
    questions = []
    for index, row in data.iterrows():
        lab_test = row['Lab Test']
        specimen = row['Specimen']
        gender_specific = row.get('Gender-specific', 'any gender').strip() if pd.notna(row.get('Gender-specific')) else 'any gender'
        age_group_specific = row.get('Age group-specific', 'any age group').strip() if pd.notna(row.get('Age group-specific')) else 'any age group'
        types_of_reference_range = row['Types of reference range']
        correct_answer = row['SI Reference Interval']

        # Option A: Correct answer
        option_a = correct_answer

        # Option B: Correct normal range from another lab test
        other_tests = data[data['Lab Test'] != lab_test]
        option_b = choice(other_tests['SI Reference Interval'].values)

        # Ensure Option B is unique
        while option_b == option_a:
            option_b = choice(other_tests['SI Reference Interval'].values)

        # Option C: Scale the correct SI reference range by multiplying with value 10
        option_c = convert_interval_to_scaled(correct_answer, 10)

        # Option D: Avoid 'nan' by ensuring a valid conversion or a valid gender-specific value
        valid_units = ['μmol/L', 'mg/dL', 'mmol/L', 'g/L']
        if row['SI Units'] in valid_units:
            option_d_raw = convert_to_different_measure(correct_answer, row['SI Units'], 'mol/L')
            option_d = format_decimal_value(option_d_raw)
            if option_d == correct_answer:  # Fallback if conversion is ineffective
                option_d_raw = gender_specific if gender_specific != 'any gender' else convert_to_different_measure(correct_answer, row['SI Units'], choice(valid_units))
                option_d = format_decimal_value(option_d_raw)
        else:
            option_d_raw = gender_specific if gender_specific != 'any gender' else convert_to_different_measure(correct_answer, 'μmol/L', 'mol/L')  # Default conversion if unit not listed
            option_d = format_decimal_value(option_d_raw)

        question = {
            "ID": index + 1,
            "Question": f"For the lab test '{lab_test}' measuring in '{row['SI Units']}' in Specimen '{specimen}' for '{gender_specific}', age group '{age_group_specific}', and types of reference range '{types_of_reference_range}', what is the correct SI reference range?",
            "Choices": f"A: {option_a}\nB: {option_b}\nC: {option_c}\nD: {option_d}",
            "Answer": "A"
        }

        questions.append(question)

    return questions

# Regenerate the questions with the decimal fix for Option D
questions_decimal_fixed = generate_questions_decimal_fix(data)

# Print the first question as a sample after fixing Option D to have only 3 decimal points
questions_decimal_fixed[:25]



# Specify the path for the JSON file where you want to save the questions
output_file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\All_A_Corpus_10Labs_Set1.json'

# Write the questions to the JSON file
with open(output_file_path, 'w') as outfile:
    json.dump(questions_decimal_fixed , outfile, indent=4)

print(f"Questions successfully saved to {output_file_path}")

#Making option 'A' as the correct answer Set 2
import json
from random import uniform, choice
import pandas as pd

# Load the Excel file
labs_df= pd.read_excel('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\10_Labs.xlsx')
def generate_questions_v2(df):
    questions = []
    id_counter = 1

    for index, row in df.iterrows():
        # Extract relevant data for each lab test
        lab_test = row['Lab Test']
        specimen = row['Specimen']
        gender_specific = ' for ‘' + row['Gender-specific'] + '’' if not pd.isnull(row['Gender-specific']) else ''
        age_group_specific = ' in age group ‘' + row['Age group-specific'] + '’' if not pd.isnull(row['Age group-specific']) else ''
        si_units = row['SI Units']
        reference_range = row['SI Reference Interval']

        # Ensure parsing for both ranges and single value limits
        if '–' in reference_range:
            low, high = reference_range.replace('<', '').replace('>', '').split('–')
            low, high = float(low), float(high)
        elif '<' in reference_range:
            high = float(reference_range.replace('<', ''))
            low = 0.1  # Adjusted to ensure value > 0
        elif '>' in reference_range:
            low = float(reference_range.replace('>', ''))
            high = low * 2  # Arbitrary upper bound for '>' scenarios
        else:
            continue  # Skip if reference range is not clear

        # Generate a positive random value for the question
        random_value = max(0.1, uniform(low - (high-low), high + (high-low)))  # Ensure value is > 0

        # Determine the correct category based on the random value
        if random_value < low:
            correct_category = 'Low'
        elif random_value > high:
            correct_category = 'High'
        else:
            correct_category = 'Normal'

        # Construct the question
        question_text = f"Q{id_counter}: For the lab test '{lab_test}' measuring in '{si_units}' in Specimen ‘{specimen}’{gender_specific}{age_group_specific} a value in 'SI reference range' is {random_value:.2f}. Is the lab test result is?"

        # Ensure the correct answer is always 'A', and the other options are different
        categories = ['Normal', 'Low', 'High']
        categories.remove(correct_category)
        options = [correct_category] + categories
        choices = "\nA: " + options[0] + "\nB: " + options[1] + "\nC: " + options[2]

        questions.append({
            "ID": id_counter,
            "Question": question_text,
            "Choices": choices,
            "Answer": "A"
        })

        id_counter += 1

    return questions

# Generate questions with the updated requirements
questions_json_v2 = generate_questions_v2(labs_df)
# Saving the updated questions as a JSON file
output_file_path_v2 = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\All_A_Corpus_10Labs_Set2.json'
with open(output_file_path_v2, 'w') as f:
    json.dump({"questions": questions_json_v2}, f, indent=4)

output_file_path_v2
questions_json_v2[:15]

#Version 3

import numpy as np
import pandas as pd

# Load the Excel sheet
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Curated556.xlsx'
data = pd.read_excel(file_path)
# Define a function to convert ranges to a consistent format of "lower-upper"
def format_range(r):
    if pd.isna(r):
        return np.nan
    if '<' in r:
        upper_bound = float(r.replace('<', '').strip())
        return f"0-{upper_bound}"
    elif '>' in r:
        lower_bound = float(r.replace('>', '').strip())
        return f"{lower_bound}-100"
    else:
        return r

# Apply the function to 'SI Reference Interval'
data['Formatted SI Range'] = data['SI Reference Interval'].apply(format_range)

# Generate questions and answers
questions = []
for idx, row in data.iterrows():
    gender_specific = 'any gender' if pd.isna(row['Gender-specific']) else row['Gender-specific']
    question = (
        f"For the lab test '{row['Lab Test']}' measuring in '{row['SI Units']}' in Specimen '{row['Specimen']}' "
        f"for '{gender_specific}', what is the correct lower and upper bound range values in SI reference range?"
    )
    answer = row['Formatted SI Range']
    questions.append({
        "ID": idx + 1,
        "Question": question,
        "Answer": answer
    })

questions[:5]  # Display the first 5 questions and answers

import json

# Path to save the JSON file
json_file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Curated556_Lab_Test_Questions.json'

# Write the list of dictionaries to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(questions, json_file, indent=4)

json_file_path  # Return the path of the saved JSON file

import json
import random

# Load the JSON data from the previously saved file
def load_questions(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Define a function to generate a random value within a specified range
def generate_random_value(lower, upper):
    return random.uniform(0.5 * lower, 2 * upper)

# Define a function to classify the generated value as 'normal', 'low', 'high'
def classify_value(value, lower, upper):
    if value < lower:
        return "Low"
    elif value > upper:
        return "High"
    else:
        return "Normal"

# Parse the answers to obtain numerical ranges
def parse_range(answer):
    # Replace non-standard dash with a standard one
    answer = answer.replace('–', '-')
    if '-' in answer:
        bounds = answer.split('-')
        lower = float(bounds[0].strip())
        upper = float(bounds[1].strip())
    else:
        lower = upper = float(answer.strip())
    return lower, upper

# Generate multiple choice questions
def create_mc_questions(lab_test_data):
    mc_questions = []
    for test in lab_test_data:
        # Parse the question to extract details
        question_parts = test['Question'].split("'")
        lab_test = question_parts[1]
        units = question_parts[5]
        specimen = question_parts[3]
        gender_specific = question_parts[7]

        lower_bound, upper_bound = parse_range(test['Answer'])
        random_value = generate_random_value(lower_bound, upper_bound)
        correct_classification = classify_value(random_value, lower_bound, upper_bound)

        choices = ["Low", "Normal", "High"]
        choices.remove(correct_classification)
        random.shuffle(choices)
        choices.append(correct_classification)
        random.shuffle(choices)
        correct_answer = "ABC"[choices.index(correct_classification)]

        question_text = (
            f": For the lab test '{lab_test}' measuring in '{units}' in Specimen '{specimen}' for '{gender_specific}', "
            f"a value in 'SI reference range' is {random_value:.2f}. Is the lab test result is?"
        )
        mc_questions.append({
            "ID": test['ID'],
            "Question": question_text,
            "Choices": "\nA: " + choices[0] + "\nB: " + choices[1] + "\nC: " + choices[2],
            "Answer": correct_answer
        })
    return mc_questions

# Save questions to JSON file
def save_questions(mc_questions, output_path):
    with open(output_path, 'w') as file:
        json.dump(mc_questions, file, indent=4)

# Main function to process and generate questions
def main():
    json_file_path='C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Curated556_Lab_Test_Questions.json' # Change to the correct path
    questions = load_questions(json_file_path)
    mc_questions = create_mc_questions(questions)
    output_json_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Curated556_Lab_Test_Questions_set2.json'  # Change to the desired output path
    save_questions(mc_questions, output_json_path)
    print(f"Questions saved to {output_json_path}")
main()

#Set 3 Evaluation
import json

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Result\\Version 3\\Lab_Test_Questions_set1_V3.json'

# Read JSON data from file
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Extract and print the Answer values
for item in json_data:
    print(item["Answer"])

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Result\\Version 3\\Lab_Test_Questions_set2_V3.json'

# Read JSON data from file
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Extract and print the Answer values
for item in json_data:
    print(item["Answer"])

import json

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Result\\Version 3\\v3_results\\v3_results\\set1_gpt-4-turbo.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers


# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Result\\Version 3\\v3_results\\v3_results\\set1_gpt-35-turbo.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers


# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Result\\Version 3\\v3_results\\v3_results\\set2_gpt-4-turbo.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers


# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Result\\Version 3\\v3_results\\v3_results\\set2_gpt-35-turbo.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers

#Set 4
import json

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Curated550_Lab_Test_Questions_set1.json'

# Read JSON data from file
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Extract and print the Answer values
for item in json_data:
    print(item["Answer"])

#Set 4
import json

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Curated550_Lab_Test_Questions_set2.json'

# Read JSON data from file
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Extract and print the Answer values
for item in json_data:
    print(item["Answer"])

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\set1_gpt-4.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\set1_gpt-35-turbo.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\set2_gpt-4.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers

# File path
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\set2_gpt-35-turbo.json'

# Read JSON data from file
try:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Please check the file content.")

# Extract and print the answer values
answers = []
try:
    for key in json_data:
        # Check if the value is a dictionary
        if isinstance(json_data[key], dict):
            answers.append(json_data[key]["answer"])
        # Handle the case where the value is a string containing JSON
        elif isinstance(json_data[key], str):
            try:
                inner_json = json.loads(json_data[key].strip('```json\n').strip('\n```'))
                answers.append(inner_json["answer"])
            except json.JSONDecodeError:
                answers.append(f"Error decoding nested JSON in entry {key}")
except KeyError as e:
    answers.append(f"Key error: {e}")
except TypeError as e:
    answers.append(f"Type error: {e}")

answers




import json
import pandas as pd
import matplotlib.pyplot as plt



# Load the JSON data
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Curated550_Lab_Test_Questions_set1.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(data)

# Split the Answer column into two separate columns for lower and upper bounds
df[['Lower_Bound', 'Upper_Bound']] = df['Answer'].str.split('-', expand=True)

# Convert the lower and upper bounds to numeric values
df['Lower_Bound'] = pd.to_numeric(df['Lower_Bound'], errors='coerce')
df['Upper_Bound'] = pd.to_numeric(df['Upper_Bound'], errors='coerce')

# Drop rows with any missing values in the bounds
df_cleaned = df.dropna(subset=['Lower_Bound', 'Upper_Bound'])

# Generate some exploratory plots

# Distribution of Lower Bound
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Lower_Bound'], bins=30, color='skyblue', alpha=0.7)
plt.title('Distribution of Lower Bound Values')
plt.xlabel('Lower Bound')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Upper bound.png', dpi=300)
plt.show()

# Distribution of Upper Bound
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Upper_Bound'], bins=30, color='salmon', alpha=0.7)
plt.title('Distribution of Upper Bound Values')
plt.xlabel('Upper Bound')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\lower bound.png', dpi=300)
plt.show()

# Scatter plot of Lower Bound vs Upper Bound
# Plot 3: Scatter Plot of Lower Bound vs Upper Bound
plt.figure(figsize=(10, 6))
plt.scatter(df['Lower_Bound'], df['Upper_Bound'], color='green', alpha=0.6)
plt.title('Scatter Plot of Lower Bound vs Upper Bound')
plt.xlabel('Lower Bound')
plt.ylabel('Upper Bound')
plt.tight_layout()
plt.grid(True)
plt.savefig('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\scatter plot.png', dpi=300)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy

# Assuming df is already created and contains the Specimen column
# Ensure all NaN values are removed
df['Specimen'] = df['Question'].str.extract(r"Specimen '([^']+)'")
df['Specimen'] = df['Specimen'].str.strip()  # Remove any leading or trailing spaces
df = df.dropna(subset=['Specimen'])  # Drop rows where Specimen is NaN

# Additional step to filter out any remaining empty strings that might have been mistaken for nan
df = df[df['Specimen'] != 'nan']

df['Specimen'] = df['Specimen'].replace({
    'Serum, plasma': 'Serum/Plasma',
    'Plasma, serum': 'Serum/Plasma',
    'Plasma or serum': 'Serum/Plasma',
    'Serum or plasma': 'Serum/Plasma',
    'Plamsa, serum': 'Serum/Plasma',  # Assuming "Plamsa" is a typo for "Plasma"
    'Serum, whole blood': 'Serum/Whole Blood',
    'Whole blood, serum': 'Serum/Whole Blood',
    'Serum, plasma, venous blood': 'Serum/Plasma/Venous Blood',
    'Arterial blood': 'Whole Blood',
    'Red blood cells': 'Whole Blood',
    'Venous blood': 'Whole Blood',
    'Blood': 'Whole Blood',
    'Serum, urine': 'Serum/Urine',
    'Urine, 24 h': 'Urine',
    'Urine 24h': 'Urine'
})

# Count the occurrences of each specimen type
specimen_counts = df['Specimen'].value_counts()

# Plot the results
plt.figure(figsize=(16, 8))
specimen_counts.plot(kind='bar', color='palevioletred', alpha=0.7)
plt.title('Number of Lab Tests per Specimen Type')
plt.xlabel('Specimen Type')
plt.ylabel('Number of Lab Tests')
plt.xticks(rotation=45, ha='right')  # Make x-axis labels horizontal
plt.tight_layout()  # Adjust layout to prevent cropping
plt.savefig('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\lab_tests_per_specimen_type.png', dpi=300)
plt.show()

# Extract short version of each lab test name from the JSON data
short_versions = []

for item in data:
    question = item.get("Question", "")
    if question:
        # Extract the lab test name which is usually the first part of the question
        short_version = question.split("'")[1] if "'" in question else question
        short_versions.append(short_version)
# Remove duplicates from the short versions list
unique_short_versions = list(set(short_versions))
# Sort the list of unique short versions of lab tests
sorted_unique_short_versions = sorted(unique_short_versions)


len(sorted_unique_short_versions)

# Display the first 20 sorted lab tests for brevity
sorted_unique_short_versions[:20]

import math

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig
plot_colortable(mcolors.CSS4_COLORS)
plt.show()

import pandas as pd
import json

# Load the Excel file
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Curated556.xlsx'
data = pd.read_excel(file_path)

# Define a function to handle missing values and format the question accordingly
def format_question(row):
    gender_specific = row.get('Gender-specific', 'any gender').strip() if pd.notna(row.get('Gender-specific')) else 'any gender'
    age_group_specific = row.get('Age group-specific', 'any age group').strip() if pd.notna(row.get('Age group-specific')) else 'any age group'
    women_condition = row.get('Women-related condition', '') if pd.notna(row.get('Women-related condition')) else ''
    category_of_lab_test = row.get('Category of Lab Test', '') if pd.notna(row.get('Category of Lab Test')) else ''

    # Skip women_condition and category_of_lab_test if they are null
    question = (
        f"For the lab test '{row['Lab Test']}' measuring in '{row['SI Units']}' in Specimen '{row['Specimen']}' "
        f"for '{gender_specific}' and '{age_group_specific}'"
    )
    if women_condition:
        question += f" with the condition '{women_condition.strip()}'"
    if category_of_lab_test:
        question += f" in the category '{category_of_lab_test.strip()}'"

    question += ", what is the correct lower and upper bound range values in SI reference range?"

    return question

# Apply the function to each row to generate the questions
data['Question'] = data.apply(format_question, axis=1)

# Generate an ID column based on the index
data['ID'] = data.index + 1

# Select relevant columns for the final output including the SI Reference Interval as the answer
output_data = data[['ID', 'Question', 'SI Reference Interval']]
output_data.rename(columns={'SI Reference Interval': 'Answer'}, inplace=True)

# Convert the DataFrame to a list of dictionaries
json_data = output_data.to_dict(orient='records')

# Save the JSON data to a new file
json_file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Curated550_Lab_Test_Questions_set3.json'
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

json_file_path


result['Question'][0]

result['Question'][25]

#Set 2
import pandas as pd
import random

# Function to generate a random value within a specified range
def generate_random_value(lower, upper):
    return random.uniform(0.5 * lower, 2 * upper)

# Function to classify the generated value as 'normal', 'low', or 'high'
def classify_value(value, lower, upper):
    if value < lower:
        return "Low"
    elif value > upper:
        return "High"
    else:
        return "Normal"

# Function to parse the answers to obtain numerical ranges
def parse_range(answer):
    answer = answer.replace('–', '-')
    if '-' in answer:
        bounds = answer.split('-')
        lower = float(bounds[0].strip())
        upper = float(bounds[1].strip())
    elif '<' in answer:
        upper = float(answer.replace('<', '').strip())
        lower = 0  # Assume lower bound as 0 when using '<'
    elif '>' in answer:
        lower = float(answer.replace('>', '').strip())
        upper = lower * 2  # Set a placeholder upper bound for '>' cases
    else:
        lower = upper = float(answer.strip())
    return lower, upper

# Function to shuffle choices and distribute correct answers evenly
def generate_questions_with_shuffled_choices(df):
    questions = []
    answer_distribution = {"A": 0, "B": 0, "C": 0}

    for index, row in df.iterrows():
        gender_specific = row.get('Gender-specific', 'any gender').strip() if pd.notna(row.get('Gender-specific')) else 'any gender'
        age_group_specific = row.get('Age group-specific', 'any age group').strip() if pd.notna(row.get('Age group-specific')) else 'any age group'
        women_condition = row.get('Women-related condition', '') if pd.notna(row.get('Women-related condition')) else ''
        category_of_lab_test = row.get('Category of Lab Test', '') if pd.notna(row.get('Category of Lab Test')) else ''

        question_text = (
            f"For the lab test '{row['Lab Test']}' measuring in '{row['SI Units']}' in Specimen '{row['Specimen']}' "
            f"for '{gender_specific}' and '{age_group_specific}'"
        )
        if women_condition:
            question_text += f" with the condition '{women_condition.strip()}'"
        if category_of_lab_test:
            question_text += f" in the category '{category_of_lab_test.strip()}'"

        si_reference_range = row['SI Reference Interval']
        lower, upper = parse_range(si_reference_range)
        value = generate_random_value(lower, upper)
        classification = classify_value(value, lower, upper)

        # Generate choices and shuffle them
        choices = ["Low", "High", "Normal"]
        random.shuffle(choices)

        # Determine the correct answer's letter (A, B, or C)
        correct_answer_letter = ["A", "B", "C"][choices.index(classification)]

        # Update the answer distribution
        if answer_distribution[correct_answer_letter] > min(answer_distribution.values()):
            # If the current choice has been used more frequently, find the least used letter
            least_used_letter = min(answer_distribution, key=answer_distribution.get)
            correct_answer_letter = least_used_letter

        # Update the distribution count
        answer_distribution[correct_answer_letter] += 1

        # Prepare the formatted choices
        formatted_choices = f"\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}"

        question_text += f", a value in 'SI reference range' is {value:.2f}. Is the lab test result?"

        question = {
            "ID": index + 1,
            "Question": question_text,
            "Choices": formatted_choices,
            "Answer": correct_answer_letter
        }

        questions.append(question)

    return questions

# Example of loading the data and generating the questions
file_path = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\\Final data set\\Curated556.xlsx'  # Update this path as necessary
df = pd.read_excel(file_path)

# Generate the questions with shuffled choices
shuffled_questions = generate_questions_with_shuffled_choices(df)

# Display the first 5 questions for review
shuffled_questions[:5]

import json

# Function to save the generated questions to a JSON file
def save_questions_to_json(questions, file_name):
    with open(file_name, 'w') as json_file:
        json.dump(questions, json_file, indent=4)

# Generate the questions with shuffled choices
shuffled_questions = generate_questions_with_shuffled_choices(df)

# Save the questions to a JSON file
json_file_name = 'C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\\Final data set\\Curated550_Lab_Test_Questions_set4.json'
save_questions_to_json(shuffled_questions, json_file_name)

# Provide the file path for downloading
json_file_name


