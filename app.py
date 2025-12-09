import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="Forecasting BBCA (Input Harga)", page_icon="📈")

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    try:
        # Load model yang sudah dilatih (pastikan file ada di satu folder)
        return joblib.load('best_model_ARIMA.joblib')
    except FileNotFoundError:
        st.error("File 'best_model_ARIMA.joblib' tidak ditemukan.")
        return None

# --- 3. Tampilan Utama ---
st.title("📈 Prediksi Harga Saham BBCA")
st.markdown("Masukkan **harga penutupan (Close)** hari ini untuk memprediksi harga besok.")

st.divider()

# --- 4. Input User (Harga Hari Ini) ---
# Kita ambil default value dari kisaran harga BBCA umum (misal 9000-10000) agar user ada gambaran
col_input, col_btn = st.columns([2, 1])

with col_input:
    current_price = st.number_input(
        "Harga Penutupan Hari Ini (Rp):", 
        min_value=0, 
        value=9500, 
        step=25,
        help="Masukkan harga saham terakhir yang terjadi hari ini."
    )

# --- 5. Logika Prediksi ---
model = load_model()

if model is not None and st.button("Prediksi Harga Besok"):
    with st.spinner('Menghitung prediksi...'):
        try:
            # A. Transformasi Logaritma (Sesuai cara training model)
            # Model dilatih dengan data np.log(), jadi input baru juga harus di-log
            current_price_log = np.log(current_price)

            # B. Update Model (Walk-Forward)
            # Kita gunakan .append() untuk menambahkan data baru ini ke history model
            # refit=False agar tidak perlu training ulang (hanya update state)
            updated_model = model.append([current_price_log], refit=False)

            # C. Forecasting 1 Hari ke Depan
            forecast_result = updated_model.get_forecast(steps=1)
            
            # Ambil nilai prediksi (Log)
            pred_log = forecast_result.predicted_mean.iloc[0]
            conf_int_log = forecast_result.conf_int().iloc[0]

            # D. Kembalikan ke Harga Asli (Inverse Log)
            pred_price = np.exp(pred_log)
            lower_bound = np.exp(conf_int_log.iloc[0])
            upper_bound = np.exp(conf_int_log.iloc[1])

            # --- 6. Menampilkan Hasil ---
            st.success("Prediksi Selesai!")
            
            # Tampilan Kartu Metrik Besar
            st.metric(
                label="Prediksi Harga Besok",
                value=f"Rp {pred_price:,.0f}",
                delta=f"{((pred_price - current_price) / current_price * 100):.2f}% (Return)",
                delta_color="normal"
            )

            # Tampilan Detail Range
            st.markdown(f"""
            **Detail Rentang Kepercayaan (95%):**
            * Batas Bawah: **Rp {lower_bound:,.0f}**
            * Batas Atas: **Rp {upper_bound:,.0f}**
            """)

            # Visualisasi Sederhana (Candlestick / Bar comparison)
            fig = go.Figure()
            
            # Bar Hari Ini
            fig.add_trace(go.Bar(
                x=['Hari Ini'], 
                y=[current_price], 
                name='Harga Hari Ini',
                marker_color='gray',
                text=[f"Rp {current_price:,.0f}"],
                textposition='auto'
            ))

            # Bar Prediksi Besok (dengan Error Bar)
            fig.add_trace(go.Bar(
                x=['Besok (Prediksi)'], 
                y=[pred_price],
                name='Prediksi',
                marker_color='blue',
                text=[f"Rp {pred_price:,.0f}"],
                textposition='auto',
                error_y=dict(
                    type='data',
                    array=[upper_bound - pred_price],
                    arrayminus=[pred_price - lower_bound],
                    visible=True
                )
            ))

            fig.update_layout(
                title="Perbandingan Hari Ini vs Besok",
                yaxis_title="Harga (IDR)",
                template="plotly_white",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.info("Tips: Pastikan input harga tidak 0 atau negatif.")

# Footer
st.markdown("---")
st.caption("Model: ARIMA Walk-Forward (Updated State)")
