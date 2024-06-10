import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
base_path = '/mnt/data'  # Adjust this path to your dataset's location
train_file = os.path.join('Kaggle Competition', 'train.csv')
test_file = os.path.join('Kaggle Competition', 'test.csv')
wetter_file = os.path.join('Kaggle Competition', 'wetter.csv')
kiwo_file = os.path.join('Kaggle Competition', 'kiwo.csv')

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
train_df = pd.merge(train_df, wetter_df, on='Datum', how='left')
test_df = pd.merge(test_df, wetter_df, on='Datum', how='left')

train_df = pd.merge(train_df, kiwo_df, on='Datum', how='left')
test_df = pd.merge(test_df, kiwo_df, on='Datum', how='left')

train_df['KielerWoche'] = train_df['KielerWoche'].fillna(0)
test_df['KielerWoche'] = test_df['KielerWoche'].fillna(0)

# Handle missing values in weather-related columns
train_df['Bewoelkung'] = train_df['Bewoelkung'].fillna(train_df['Bewoelkung'].mean())
train_df['Temperatur'] = train_df['Temperatur'].fillna(train_df['Temperatur'].mean())
train_df['Windgeschwindigkeit'] = train_df['Windgeschwindigkeit'].fillna(train_df['Windgeschwindigkeit'].mean())
train_df['Wettercode'] = train_df['Wettercode'].fillna(train_df['Wettercode'].mode()[0])

test_df['Bewoelkung'] = test_df['Bewoelkung'].fillna(test_df['Bewoelkung'].mean())
test_df['Temperatur'] = test_df['Temperatur'].fillna(test_df['Temperatur'].mean())
test_df['Windgeschwindigkeit'] = test_df['Windgeschwindigkeit'].fillna(test_df['Windgeschwindigkeit'].mean())
test_df['Wettercode'] = test_df['Wettercode'].fillna(test_df['Wettercode'].mode()[0])

# Feature engineering: Extract additional features from the date column
train_df['Year'] = train_df['Datum'].dt.year
train_df['Month'] = train_df['Datum'].dt.month
train_df['Day'] = train_df['Datum'].dt.day
train_df['DayOfWeek'] = train_df['Datum'].dt.dayofweek

test_df['Year'] = test_df['Datum'].dt.year
test_df['Month'] = test_df['Datum'].dt.month
test_df['Day'] = test_df['Datum'].dt.day
test_df['DayOfWeek'] = test_df['Datum'].dt.dayofweek

# Set the plot style
sns.set(style="whitegrid")

# Visualize sales trends over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=train_df, x='Datum', y='Umsatz', hue='Warengruppe', palette='tab10')
plt.title('Sales Trends Over Time by Product Category')
plt.xlabel('Date')
plt.ylabel('Sales (Umsatz)')
plt.legend(title='Product Category', loc='upper right')
plt.savefig(os.path.join('Kaggle Competition', 'sales_trends_over_time.png'))

# Analyze the impact of temperature on sales
plt.figure(figsize=(14, 7))
sns.scatterplot(data=train_df, x='Temperatur', y='Umsatz', hue='Warengruppe', palette='tab10')
plt.title('Impact of Temperature on Sales by Product Category')
plt.xlabel('Temperature (°C)')
plt.ylabel('Sales (Umsatz)')
plt.legend(title='Product Category', loc='upper right')
plt.savefig(os.path.join('Kaggle Competition', 'impact_of_temperature_on_sales.png'))

# Analyze the impact of "Kieler Woche" on sales
plt.figure(figsize=(14, 7))
sns.boxplot(data=train_df, x='KielerWoche', y='Umsatz', hue='Warengruppe', palette='tab10')
plt.title('Impact of Kieler Woche on Sales by Product Category')
plt.xlabel('Kieler Woche (1 = During, 0 = Not During)')
plt.ylabel('Sales (Umsatz)')
plt.legend(title='Product Category', loc='upper right')
plt.savefig(os.path.join('Kaggle Competition', 'impact_of_kieler_woche_on_sales.png'))

