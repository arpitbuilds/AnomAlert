import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load models with caching
@st.cache_resource
def load_models():
    rf = joblib.load('failure_rf_model.pkl')
    iso = joblib.load('isolation_forest_model.pkl')
    return rf, iso

rf_model, iso_model = load_models()

# 2. App title and sidebar
st.set_page_config(page_title="AnomAlert", layout="wide")
st.title("🚨 AnomAlert - Industrial Monitoring Dashboard")

# 3. Tabs for modular navigation
tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical Analysis", "🔍 Real-Time Monitoring", "⚙️ Retrain Model", "📥 Alert Logs"])

# TAB 1: Historical Analysis
with tab1:
    uploaded_file = st.file_uploader("Upload Historical Sensor Data CSV", type="csv")

    if uploaded_file:
        hist_df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
        st.subheader("Sensor Data Overview")
        st.dataframe(hist_df.tail())

        st.subheader("📊 Sensor Trends Over Time")
        st.line_chart(hist_df.set_index('timestamp')[['temperature', 'vibration', 'pressure', 'gas_ppm']])

        hist_features = hist_df[['temperature', 'vibration', 'pressure', 'gas_ppm']]
        hist_df['failure_pred'] = rf_model.predict(hist_features)
        hist_df['anomaly_pred'] = iso_model.predict(hist_features)
        hist_df['anomaly_label'] = hist_df['anomaly_pred'].apply(lambda x: "Anomaly" if x == -1 else "Normal")

        st.subheader("⚠️ Failure Predictions")
        st.bar_chart(hist_df.set_index('timestamp')['failure_pred'])

        st.subheader("🧪 Anomaly Detection")
        st.bar_chart(hist_df.set_index('timestamp')['anomaly_pred'])

        st.subheader("🎞️ Simulated Streaming")
        if st.button("▶️ Start Simulation"):
            for i in range(len(hist_df)):
                row = hist_df.iloc[i]
                st.write(f"🕒 {row['timestamp']} | 🌡️ Temp: {row['temperature']}°C | 💥 Failure: {'Yes' if row['failure_pred']==1 else 'No'} | 🔍 {row['anomaly_label']}")
                time.sleep(0.5)

# TAB 2: Real-Time Monitoring
with tab2:
    st.subheader(" — Test Now!")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temperature = st.slider('Temperature (°C)', 50.0, 100.0, 70.0)
    with col2:
        vibration = st.slider('Vibration (mm/s)', 1.0, 6.0, 3.0)
    with col3:
        pressure = st.slider('Pressure (bar)', 3.0, 7.0, 5.0)
    with col4:
        gas_ppm = st.slider('Gas PPM', 100, 300, 200)

    input_data = pd.DataFrame({
        'temperature': [temperature],
        'vibration': [vibration],
        'pressure': [pressure],
        'gas_ppm': [gas_ppm]
    })

    failure_pred = rf_model.predict(input_data)[0]
    failure_prob = rf_model.predict_proba(input_data)[0][1]
    anomaly_pred = iso_model.predict(input_data)[0]
    anomaly_label = "Anomaly Detected 🚨" if anomaly_pred == -1 else "Normal ✅"

    colA, colB = st.columns(2)
    with colA:
        st.metric(label="Failure Prediction", value="Yes" if failure_pred else "No", delta=f"{failure_prob:.2f}")
    with colB:
        st.metric(label="Anomaly Detection", value=anomaly_label)

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

# TAB 3: Retrain Model
with tab3:
    st.subheader("🔁 Upload New Labeled Data")
    new_data_file = st.file_uploader("Upload CSV with Labels (for Retraining)", type="csv", key="retrain")
    if new_data_file:
        new_df = pd.read_csv(new_data_file)
        if all(col in new_df.columns for col in ['temperature', 'vibration', 'pressure', 'gas_ppm', 'failure']):
            X_new = new_df[['temperature', 'vibration', 'pressure', 'gas_ppm']]
            y_new = new_df['failure']
            rf_model.fit(X_new, y_new)
            joblib.dump(rf_model, 'failure_rf_model_updated.pkl')
            st.success("✅ Model retrained and saved as 'failure_rf_model_updated.pkl'")
        else:
            st.error("CSV must include: temperature, vibration, pressure, gas_ppm, failure")

    st.subheader("📈 Model Evaluation")
    try:
        test_df = pd.read_csv('sensor_data.csv')
        X_test = test_df[['temperature', 'vibration', 'pressure', 'gas_ppm']]
        y_test = test_df['failure']
        y_pred_test = rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test)
        rec = recall_score(y_test, y_pred_test)
        st.write(f"🔵 Accuracy: **{acc:.2f}** | 🟢 Precision: **{prec:.2f}** | 🔴 Recall: **{rec:.2f}**")

        feat_imp_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': rf_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        st.bar_chart(feat_imp_df.set_index('feature'))
    except Exception as e:
        st.info("Model performance metrics unavailable.")

# TAB 4: Alert Logs
with tab4:
    st.subheader("🚨 Your Critical Alert Log — Take Action Now!")
    if 'alerts' in st.session_state and st.session_state['alerts']:
        alerts_df = pd.DataFrame(st.session_state['alerts'])
        st.dataframe(alerts_df)
        csv = alerts_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Alert Log CSV", data=csv, file_name='alert_log.csv', mime='text/csv')
    else:
        st.info("No alerts recorded yet.")
