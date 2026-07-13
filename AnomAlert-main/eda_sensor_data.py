import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('sensor_data.csv')

# Quick overview
print("Dataset shape:", df.shape)
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Summary statistics
print("\nSummary statistics:\n", df.describe())

# Distribution plots for sensors
sensor_cols = ['temperature', 'vibration', 'pressure', 'gas_ppm']

plt.figure(figsize=(12,8))
for i, col in enumerate(sensor_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')

plt.title('Correlation Matrix')
plt.show()

# Failure vs sensor values boxplots
plt.figure(figsize=(12,8))
for i, col in enumerate(sensor_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='failure', y=col, data=df)
    plt.title(f'{col} vs Failure')
plt.tight_layout()
plt.show()
