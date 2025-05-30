#GPT results
import os
import openai
#api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = 'OPENAI_API_KEY'

# Load the dataset
import json
file_path = json.load(open("C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Curated550_Lab_Test_Questions_set1.json"))
# Function to load JSON data safely
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)  # Use json.load instead of json.loads
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load the data
dataset = load_json_file(json_file_path)

# Check if the data loaded successfully
dataset[:5] if loaded_data else "Failed to load JSON data."

results_list = []


# Loop over each entry in the dataset to process the question
for entry in dataset:
    question = entry["Question"]
    context = ""  # If you have any additional context, include it here.
    prompt = f"{context}\n\nBased on the above information, {question}"

    # Query GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant. Your task is to answer questions "
                    "related to lab tests. You should recall the reference range if that is not provided. "
                    "Provide the answer in the format [answer: lower range-upper range], Example [answer: 10-350]."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    output = response['choices'][0]['message']['content']

    # Append the result to the list
    results_list.append({'ID': entry['ID'], 'GPT-4o Output': output})

# Convert the results to a DataFrame and display it
df_results = pd.DataFrame(results_list)
print(df_results.to_string(index=False))

df_results.to_csv('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Set1_GPT_4o.csv')

results_list = []
# Loop over each entry in the dataset to process the question
for entry in dataset:
    question = entry["Question"]
    context = ""  # If you have any additional context, include it here.
    prompt = f"{context}\n\nBased on the above information, {question}"

    # Query GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant. Your task is to answer questions "
                    "related to lab tests. You should recall the reference range if that is not provided. "
                    "Provide the answer in the format [answer: lower range-upper range], Example [answer: 10-350]."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    output = response['choices'][0]['message']['content']

    # Append the result to the list
    results_list.append({'ID': entry['ID'], 'GPT-4 turbo Output': output})

# Convert the results to a DataFrame and display it
df_results = pd.DataFrame(results_list)
df_results.to_csv('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Set1_GPT_4.csv')
print(df_results.to_string(index=False))

results_list = []
# Loop over each entry in the dataset to process the question
for entry in dataset:
    question = entry["Question"]
    context = ""  # If you have any additional context, include it here.
    prompt = f"{context}\n\nBased on the above information, {question}"

    # Query GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant. Your task is to answer questions "
                    "related to lab tests. You should recall the reference range if that is not provided. "
                    "Provide the answer in the format [answer: lower range-upper range], Example [answer: 10-350]."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    output = response['choices'][0]['message']['content']

    # Append the result to the list
    results_list.append({'ID': entry['ID'], 'GPT-3.5 Output': output})

# Convert the results to a DataFrame and display it
df_results = pd.DataFrame(results_list)
df_results.to_csv('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Set1_GPT_3.5.csv')
print(df_results.to_string(index=False))

#Set 2
# Load the dataset
import json
file_path = json.load(open("C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Curated550_Lab_Test_Questions_set4.json"))
# Function to load JSON data safely
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)  # Use json.load instead of json.loads
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load the data
dataset = load_json_file(json_file_path)

# Check if the data loaded successfully
dataset[:5] if loaded_data else "Failed to load JSON data."

results_list = []

# Loop over each entry in the dataset to process the question
for entry in dataset:
    question = entry["Question"]
    context = ""  # If you have any additional context, include it here.
    prompt = f"{context}\n\nBased on the above information, {question}"

    # Query GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant. Your task is to answer questions "
                    "related to lab tests. You should recall the reference range if that is not provided. "
                    "Provide the answer in the format [A | B| C], Example [B]."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    output = response['choices'][0]['message']['content']

    # Append the result to the list
    results_list.append({'ID': entry['ID'], 'GPT-4o Output': output})

# Convert the results to a DataFrame and display it
df_results = pd.DataFrame(results_list)
df_results.to_csv('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Set2_GPT_4o.csv')
print(df_results.to_string(index=False))

# Loop over each entry in the dataset to process the question
for entry in dataset:
    question = entry["Question"]
    context = ""  # If you have any additional context, include it here.
    prompt = f"{context}\n\nBased on the above information, {question}"

    # Query GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant. Your task is to answer questions "
                    "related to lab tests. You should recall the reference range if that is not provided. "
                    "Provide the answer in the format [A | B| C], Example [B]."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    output = response['choices'][0]['message']['content']

    # Append the result to the list
    results_list.append({'ID': entry['ID'], 'GPT-4 Output': output})

# Convert the results to a DataFrame and display it
df_results = pd.DataFrame(results_list)
df_results.to_csv('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Set2_GPT_4.csv')
print(df_results.to_string(index=False))

# Loop over each entry in the dataset to process the question
for entry in dataset:
    question = entry["Question"]
    context = ""  # If you have any additional context, include it here.
    prompt = f"{context}\n\nBased on the above information, {question}"

    # Query GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful medical assistant. Your task is to answer questions "
                    "related to lab tests. You should recall the reference range if that is not provided. "
                    "Provide the answer in the format [A | B| C], Example [B]."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    output = response['choices'][0]['message']['content']

    # Append the result to the list
    results_list.append({'ID': entry['ID'], 'GPT-3.5 Output': output})

# Convert the results to a DataFrame and display it
df_results = pd.DataFrame(results_list)
df_results.to_csv('C:\\D\\e Health Lab projects\\Question_Answering\\Lab test corpus\\Final data set\\Set2_GPT_3.5.csv')
print(df_results.to_string(index=False))

