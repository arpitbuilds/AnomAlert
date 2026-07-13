import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load models
rf_model = joblib.load('failure_rf_model.pkl')
iso_model = joblib.load('isolation_forest_model.pkl')

st.title("AnomAlert — Know Before It Breaks")

# Sidebar for historical data upload
uploaded_file = st.sidebar.file_uploader("Upload Historical Sensor Data CSV", type="csv")

if uploaded_file:
    hist_df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    st.subheader("📈 Historical Sensor Data")
    st.line_chart(hist_df.set_index('timestamp')[['temperature', 'vibration', 'pressure', 'gas_ppm']])

    # Predict failure and anomaly on historical data
    hist_features = hist_df[['temperature', 'vibration', 'pressure', 'gas_ppm']]
    hist_df['failure_pred'] = rf_model.predict(hist_features)
    hist_df['anomaly_pred'] = iso_model.predict(hist_features)
    hist_df['anomaly_label'] = hist_df['anomaly_pred'].apply(lambda x: "Anomaly" if x == -1 else "Normal")

    st.subheader("⚠️ Failure Predictions Over Time")
    st.bar_chart(hist_df.set_index('timestamp')['failure_pred'])

    st.subheader("⚠️ Anomaly Detection Over Time")
    st.bar_chart(hist_df.set_index('timestamp')['anomaly_pred'])

# User inputs for real-time prediction
st.subheader("🔍 Real-Time Sensor  — Test Now!")

temperature = st.slider('Temperature (°C)', 50.0, 100.0, 70.0)
vibration = st.slider('Vibration (mm/s)', 1.0, 6.0, 3.0)
pressure = st.slider('Pressure (bar)', 3.0, 7.0, 5.0)
gas_ppm = st.slider('Gas Concentration (ppm)', 100, 300, 200)

input_data = pd.DataFrame({
    'temperature': [temperature],
    'vibration': [vibration],
    'pressure': [pressure],
    'gas_ppm': [gas_ppm]
})

# Predictions
failure_pred = rf_model.predict(input_data)[0]
failure_prob = rf_model.predict_proba(input_data)[0][1]
anomaly_pred = iso_model.predict(input_data)[0]
anomaly_label = "Anomaly Detected 🚨" if anomaly_pred == -1 else "Normal"

# Threshold-based alerting and alert log
ALERT_THRESHOLD = 0.7
if failure_prob > ALERT_THRESHOLD:
    st.warning("⚠️ High failure risk detected!")

    if 'alerts' not in st.session_state:
        st.session_state['alerts'] = []
    st.session_state['alerts'].append({
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'gas_ppm': gas_ppm,
        'failure_prob': round(failure_prob, 3)
    })

if 'alerts' in st.session_state and st.session_state['alerts']:
    st.subheader("🚨 Your Critical Alert Log — Take Action Now!")
    alerts_df = pd.DataFrame(st.session_state['alerts'])
    st.table(alerts_df)

# Display predictions
st.markdown("### What we say:")
st.write(f"Failure Prediction: {'Yes' if failure_pred == 1 else 'No'}")
st.write(f"Failure Probability: {failure_prob:.2f}")
st.write(f"Anomaly Detection: {anomaly_label}")

# Model performance and feature importance display (precomputed or on the fly)
st.subheader("📊 Model Performance & Feature Importance")

# Dummy test set metrics for display - you can replace with real test data if available
# Here we assume test set loaded from CSV named 'test_sensor_data.csv' for demo
try:
    test_df = pd.read_csv('sensor_data.csv')
    X_test = test_df[['temperature', 'vibration', 'pressure', 'gas_ppm']]
    y_test = test_df['failure']
    y_pred_test = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"Precision: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")

    # Feature importance
    importances = rf_model.feature_importances_
    features = X_test.columns
    feat_imp_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)
    st.bar_chart(feat_imp_df.set_index('feature'))
except Exception as e:
    st.write("Model performance data unavailable:", e)



st.subheader(" Make the Model Smarter !")

new_data_file = st.file_uploader("Upload New Labeled Data CSV", type="csv", key="retrain")
if new_data_file:
    new_df = pd.read_csv(new_data_file)
    if all(col in new_df.columns for col in ['temperature', 'vibration', 'pressure', 'gas_ppm', 'failure']):
        X_new = new_df[['temperature', 'vibration', 'pressure', 'gas_ppm']]
        y_new = new_df['failure']
        rf_model.fit(X_new, y_new)
        joblib.dump(rf_model, 'failure_rf_model_updated.pkl')
        st.success("✅ Model retrained and saved as 'failure_rf_model_updated.pkl'")
    else:
        st.error("Uploaded data must have columns: temperature, vibration, pressure, gas_ppm, failure")


if 'alerts' in st.session_state and st.session_state['alerts']:
    alerts_df = pd.DataFrame(st.session_state['alerts'])
    csv = alerts_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Alert Log CSV", data=csv, file_name='alert_log.csv', mime='text/csv')

