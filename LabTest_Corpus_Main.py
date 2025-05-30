import pandas as pd
import random
import json

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
    json_file_path='file_path' # Change to the correct path
    questions = load_questions(json_file_path)
    mc_questions = create_mc_questions(questions)
    output_json_path = 'file_path'  # Change to the desired output path
    save_questions(mc_questions, output_json_path)
    print(f"Questions saved to {output_json_path}")
main()

#Set 2
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

