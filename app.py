import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import date, timedelta

# 1. Setup Halaman
st.set_page_config(page_title="Forecasting BBCA", page_icon="📈")

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('best_model_ARIMA.joblib')

# 3. Tampilan Utama & Input
st.title("📈 Prediksi Saham BBCA (ARIMA)")

# Input langsung di halaman utama (lebih simpel daripada sidebar)
col1, col2 = st.columns(2)
n_days = col1.number_input("Jumlah Hari", 1, 365, 7)
start_date = col2.date_input("Mulai Tanggal", date.today() + timedelta(days=1))

# 4. Logika Prediksi
if st.button("Mulai Prediksi"):
    try:
        model = load_model()
        
        # A. Lakukan Forecasting (Baris ini yang hilang di kode Anda sebelumnya)
        forecast_result = model.get_forecast(steps=n_days)
        
        # B. Balikkan Logaritma ke Harga Asli
        pred_price = np.exp(forecast_result.predicted_mean)
        conf_int = np.exp(forecast_result.conf_int())

        # C. Buat DataFrame
        df = pd.DataFrame({
            'Tanggal': pd.date_range(start=start_date, periods=n_days, freq='B'),
            'Prediksi': pred_price.values,
            'Batas Bawah': conf_int.iloc[:, 0].values,
            'Batas Atas': conf_int.iloc[:, 1].values
        })

        # D. Tampilkan Grafik (Plotly yang disederhanakan)
        fig = go.Figure([
            # Area Arsir (Confidence Interval)
            go.Scatter(x=df['Tanggal'], y=df['Batas Atas'], line=dict(width=0), showlegend=False),
            go.Scatter(x=df['Tanggal'], y=df['Batas Bawah'], fill='tonexty', 
                       fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), name='Rentang 95%'),
            # Garis Prediksi
            go.Scatter(x=df['Tanggal'], y=df['Prediksi'], mode='lines+markers', name='Harga Prediksi')
        ])
        st.plotly_chart(fig, use_container_width=True)

        # E. Tampilkan Tabel
        st.write("Detail Angka:")
        st.dataframe(df.style.format("Rp {:,.0f}"))

    except Exception as e:
        st.error(f"Error: {e}. Pastikan file .joblib ada.")
