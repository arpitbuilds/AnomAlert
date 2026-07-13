import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Number of records
num_records = 1000

# Start time
start_time = datetime.now()

# Data generation
data = {
    'timestamp': [start_time + timedelta(seconds=i*30) for i in range(num_records)],
    'temperature': np.random.normal(loc=70, scale=5, size=num_records),
    'vibration': np.random.normal(loc=3, scale=0.5, size=num_records),
    'pressure': np.random.normal(loc=5, scale=0.3, size=num_records),
    'gas_ppm': np.random.normal(loc=200, scale=20, size=num_records)
}

df = pd.DataFrame(data)

# Failure logic: if any value crosses a threshold, label it 1
df['failure'] = df.apply(lambda row: 1 if (
    row['temperature'] > 80 or
    row['vibration'] > 4 or
    row['pressure'] > 5.5 or
    row['gas_ppm'] > 250
) else 0, axis=1)

# Save to CSV
df.to_csv('sensor_data.csv', index=False)

print("✅ Sensor data generated and saved to 'sensor_data.csv'")
