import os
import pandas as pd

# Define the file paths using os
base_path = '/mnt/data'  # Adjust this path to your dataset's location
train_file = os.path.join('Kaggle Competition', 'train.csv')
test_file = os.path.join('Kaggle Competition', 'test.csv')
wetter_file = os.path.join('Kaggle Competition', 'wetter.csv')
kiwo_file = os.path.join('Kaggle Competition', 'kiwo.csv')

# Load the datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
wetter_df = pd.read_csv(wetter_file)
kiwo_df = pd.read_csv(kiwo_file)

# Convert date columns to datetime format
train_df['Datum'] = pd.to_datetime(train_df['Datum'])
test_df['Datum'] = pd.to_datetime(test_df['Datum'])
wetter_df['Datum'] = pd.to_datetime(wetter_df['Datum'])
kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])

# Merge the datasets
# Merge weather data with train and test data
train_df = pd.merge(train_df, wetter_df, on='Datum', how='left')
test_df = pd.merge(test_df, wetter_df, on='Datum', how='left')

# Merge Kieler Woche data with train and test data
train_df = pd.merge(train_df, kiwo_df, on='Datum', how='left')
test_df = pd.merge(test_df, kiwo_df, on='Datum', how='left')

# Fill NaN values in KielerWoche with 0 (assuming days not in Kieler Woche have 0)
train_df['KielerWoche'] = train_df['KielerWoche'].fillna(0)
test_df['KielerWoche'] = test_df['KielerWoche'].fillna(0)

# Handling missing values in weather-related columns
# Fill missing values in Bewoelkung, Temperatur, and Windgeschwindigkeit with the mean of the respective columns
train_df['Bewoelkung'].fillna(train_df['Bewoelkung'].mean(), inplace=True)
train_df['Temperatur'].fillna(train_df['Temperatur'].mean(), inplace=True)
train_df['Windgeschwindigkeit'].fillna(train_df['Windgeschwindigkeit'].mean(), inplace=True)
train_df['Wettercode'].fillna(train_df['Wettercode'].mode()[0], inplace=True)  # Use mode for categorical weather code

test_df['Bewoelkung'].fillna(test_df['Bewoelkung'].mean(), inplace=True)
test_df['Temperatur'].fillna(test_df['Temperatur'].mean(), inplace=True)
test_df['Windgeschwindigkeit'].fillna(test_df['Windgeschwindigkeit'].mean(), inplace=True)
test_df['Wettercode'].fillna(test_df['Wettercode'].mode()[0], inplace=True)

# Feature engineering: Extract additional features from the date column
train_df['Year'] = train_df['Datum'].dt.year
train_df['Month'] = train_df['Datum'].dt.month
train_df['Day'] = train_df['Datum'].dt.day
train_df['DayOfWeek'] = train_df['Datum'].dt.dayofweek

test_df['Year'] = test_df['Datum'].dt.year
test_df['Month'] = test_df['Datum'].dt.month
test_df['Day'] = test_df['Datum'].dt.day
test_df['DayOfWeek'] = test_df['Datum'].dt.dayofweek

# Display the first few rows of the modified training dataset to verify the changes
print(train_df.head())
print(test_df.head())