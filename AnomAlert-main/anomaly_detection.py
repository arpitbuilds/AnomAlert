import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load data
df = pd.read_csv('sensor_data.csv')

# Features only (unsupervised)
X = df[['temperature', 'vibration', 'pressure', 'gas_ppm']]

# Initialize model
iso_forest = IsolationForest(contamination=0.1, random_state=42)

# Train
iso_forest.fit(X)

# Predict anomalies (-1 = anomaly, 1 = normal)
pred = iso_forest.predict(X)

# Convert to 0 (normal) and 1 (anomaly)
df['anomaly'] = [1 if x == -1 else 0 for x in pred]

# Evaluation (against known failures)
print("Accuracy:", accuracy_score(df['failure'], df['anomaly']))
print("\nClassification Report:\n", classification_report(df['failure'], df['anomaly']))
print("\nConfusion Matrix:\n", confusion_matrix(df['failure'], df['anomaly']))

# Save model
joblib.dump(iso_forest, 'isolation_forest_model.pkl')
print("✅ Isolation Forest model saved as isolation_forest_model.pkl")
