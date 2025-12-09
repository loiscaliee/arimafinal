import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(
    page_title="Forecasting Saham BBCA",
    page_icon="📈",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        # Pastikan file .joblib ada di folder yang sama
        model = joblib.load('best_model_ARIMA.joblib')
        return model
    except FileNotFoundError:
        st.error("File 'best_model_ARIMA.joblib' tidak ditemukan. Pastikan file model sudah diupload.")
        return None

st.title("📈 Prediksi Harga Saham BBCA")
st.markdown("""
Aplikasi ini menggunakan model **ARIMA (Walk-Forward)** untuk memprediksi harga penutupan saham BBCA.
Model dilatih menggunakan data historis yang telah di-transformasi Logaritma.
""")

st.divider()


st.sidebar.header("Pengaturan Prediksi")


n_days = st.sidebar.number_input(
    "Jumlah Hari Prediksi:",
    min_value=1,
    max_value=365,
    value=7,
    step=1,
    help="Berapa hari ke depan yang ingin Anda ramal?"
)


start_date = st.sidebar.date_input(
    "Mulai Prediksi dari Tanggal:",
    value=date.today() + timedelta(days=1),
    help="Pilih tanggal awal untuk hasil prediksi (biasanya hari esok)."
)

model = load_model()

if model is not None and st.button("Mulai Prediksi"):
    with st.spinner('Sedang melakukan forecasting...'):
        try:
           
            
            forecast_log = forecast_result.predicted_mean
            conf_int_log = forecast_result.conf_int()

            forecast_price = np.exp(forecast_log)
            conf_int_price = np.exp(conf_int_log)
           
            date_range = pd.date_range(start=start_date, periods=n_days, freq='B')
            
            df_result = pd.DataFrame({
                'Tanggal': date_range,
                'Prediksi (IDR)': forecast_price.values,
                'Batas Bawah': conf_int_price.iloc[:, 0].values,
                'Batas Atas': conf_int_price.iloc[:, 1].values
            })

      
            col1, col2 = st.columns(2)
            col1.metric("Harga Prediksi Terakhir", f"Rp {df_result['Prediksi (IDR)'].iloc[-1]:,.0f}")
            col2.metric("Trend (H+1 ke H+Akhir)", f"{((df_result['Prediksi (IDR)'].iloc[-1] - df_result['Prediksi (IDR)'].iloc[0]) / df_result['Prediksi (IDR)'].iloc[0] * 100):.2f}%")

   
            fig = go.Figure()

    
            fig.add_trace(go.Scatter(
                x=df_result['Tanggal'].tolist() + df_result['Tanggal'].tolist()[::-1],
                y=df_result['Batas Atas'].tolist() + df_result['Batas Bawah'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Confidence Interval (95%)'
            ))

   
            fig.add_trace(go.Scatter(
                x=df_result['Tanggal'],
                y=df_result['Prediksi (IDR)'],
                mode='lines+markers',
                name='Prediksi Harga',
                line=dict(color='blue', width=3)
            ))

            fig.update_layout(
                title="Forecast Pergerakan Harga BBCA",
                xaxis_title="Tanggal",
                yaxis_title="Harga Saham (IDR)",
                template="plotly_white",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

          
            st.subheader("Data Hasil Prediksi")
            st.dataframe(
                df_result.style.format({
                    "Prediksi (IDR)": "Rp {:,.2f}",
                    "Batas Bawah": "Rp {:,.2f}",
                    "Batas Atas": "Rp {:,.2f}"
                })
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat forecasting: {e}")
            st.warning("Tips: Pastikan versi statsmodels di requirements.txt sama dengan versi saat training.")

st.markdown("---")
st.caption("Dibuat dengan Streamlit & Statsmodels • Model: ARIMA Walk-Forward")
