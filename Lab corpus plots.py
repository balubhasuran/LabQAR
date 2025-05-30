#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy


# In[2]:


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


# In[9]:


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
    'Serum, whole blood': 'Whole Blood',
    'Whole blood, serum': 'Whole Blood',
    'Serum, plasma, venous blood': 'Serum/Plasma',
    'Arterial blood': 'Whole Blood',
    'Red blood cells': 'Whole Blood',
    'Venous blood': 'Whole Blood',
    'Whole blood': 'Whole Blood',
    'Blood': 'Whole Blood',
    'Serum, urine': 'Urine',
    'Urine, 24 h': 'Urine',
    'Urine 24h': 'Urine',
    'Cerebrospinal fluid': 'Other Specimen Types',  # Group under 'Other Specimen Types'
    'Stool': 'Other Specimen Types'                # Group under 'Other Specimen Typ
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


# In[ ]:




