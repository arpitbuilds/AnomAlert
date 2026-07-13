
# AnomAlert — Know Before It Breaks 🚨

AnomAlert is a real-time dashboard for detecting sensor anomalies and predicting equipment failure. It leverages machine learning models (Random Forest for failure prediction and Isolation Forest for anomaly detection) to alert users and visualize live sensor input.

## 🔧 Features

- 📊 Real-time sensor input analysis
- ⚠️ Live failure prediction and anomaly detection
- 📝 Alert log with timestamped predictions
- 📈 Historical data visualization
- 🔁 Model retraining with new labeled data
- 📥 Downloadable alert logs

## 🛠️ Setup & Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/AnomAlert.git
cd AnomAlert
```

### 2. Create and activate a virtual environment

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Place the trained models

Make sure the following files are in the root directory:
- `failure_rf_model.pkl`
- `isolation_forest_model.pkl`

### 5. Run the Streamlit app

```
streamlit run app.py
```

## 📦 File Structure

```
AnomAlert/
│
├── app.py                       # Streamlit application
├── failure_rf_model.pkl        # Pre-trained Random Forest model
├── isolation_forest_model.pkl  # Pre-trained Isolation Forest model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 📄 License

MIT License
