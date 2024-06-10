import matplotlib.pyplot as plt
import os

# Model names and their MAPE values
model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Neural Network']
mape_values = [0.820707310796834, 0.23627310658321882, 0.172167411743859, 0.8134415866572539]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(model_names, mape_values, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Performance Comparison (MAPE)')
plt.xlabel('Model')
plt.ylabel('MAPE')
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1 for better visualization
plt.grid(axis='y')

# Save the figure as a PNG file
plt.savefig(os.path.join('Kaggle Competition', 'model_performance_comparison.png'))

# Show the plot
plt.show()
