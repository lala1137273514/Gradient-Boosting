import pandas as pd

# Load the dataset
file_path = '/mnt/data/Labelled_Test_Cases.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()

# Attempt to load the dataset with a different encoding
data = pd.read_csv(file_path, encoding='ISO-8859-1')
data.head()

# Clean the dataset by selecting relevant columns and removing NaN
data = data[['v1', 'v2']].dropna()
data.columns = ['Label', 'Text']
data.head()
