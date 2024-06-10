import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load and prepare the datasets
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
def extract_date_features(df):
    df['Year'] = df['Datum'].dt.year
    df['Month'] = df['Datum'].dt.month
    df['Day'] = df['Datum'].dt.day
    df['DayOfWeek'] = df['Datum'].dt.dayofweek
    df['WeekOfYear'] = df['Datum'].dt.isocalendar().week
    df['Quarter'] = df['Datum'].dt.quarter
    return df

train_df = extract_date_features(train_df)
test_df = extract_date_features(test_df)

# Define features and target variable
features = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode', 
            'KielerWoche', 'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'Quarter', 'Warengruppe']
target = 'Umsatz'

X = train_df[features]
y = train_df[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    print(f'{model_name} MAPE: {mape}')
    return mape

# Train and evaluate models
mape_lr = train_and_evaluate_model(LinearRegression(), 'Baseline Linear Regression')
mape_dt = train_and_evaluate_model(DecisionTreeRegressor(random_state=42), 'Decision Tree Regressor')
mape_rf = train_and_evaluate_model(RandomForestRegressor(random_state=42, n_estimators=100), 'Random Forest Regressor')
mape_mlp = train_and_evaluate_model(MLPRegressor(random_state=42, max_iter=1000), 'Neural Network (MLP Regressor)')

# Model names and their MAPE values
model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Neural Network']
mape_values = [mape_lr, mape_dt, mape_rf, mape_mlp]

# Create a bar chart to visualize MAPE values
plt.figure(figsize=(10, 6))
plt.bar(model_names, mape_values, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Performance Comparison (MAPE)')
plt.xlabel('Model')
plt.ylabel('MAPE')
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1 for better visualization
plt.grid(axis='y')

# Save the figure as a PNG file
plt.savefig(os.path.join('Kaggle Competition', 'model_performance_comparison.png'))

# No plt.show() here as we are saving the plot
