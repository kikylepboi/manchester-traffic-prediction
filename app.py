import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- KONFIGURASI ---
st.set_page_config(page_title="Manchester Traffic Prediction", layout="wide")

# --- FUNGSI LOAD ASSETS (SANGAT PENTING) ---
@st.cache_resource
def load_assets():
    # Cek apakah file ada sebelum di-load
    if os.path.exists('model_traffic_manchester.pkl') and os.path.exists('label_encoder.pkl'):
        model = joblib.load('model_traffic_manchester.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, le
    else:
        return None, None

model, le = load_assets()

# --- TAMPILAN ---
st.title("🚗 Sistem Prediksi Arus Lalu Lintas Manchester")

if model is None or le is None:
    st.error("❌ ERROR: File model atau encoder tidak ditemukan!")
    st.info("Pastikan file 'model_traffic_manchester.pkl' dan 'label_encoder.pkl' sudah di-upload ke GitHub di folder yang sama dengan app.py")
else:
    st.sidebar.header("Input Parameter")
    
    # Input dari user
    detid = st.sidebar.selectbox("Pilih ID Detektor", le.classes_)
    day = st.sidebar.selectbox("Pilih Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
    hour = st.sidebar.slider("Jam", 0, 23, 12)
    occ = st.sidebar.number_input("Occupancy (Kepadatan)", 0.0, 100.0, 10.0)
    speed = st.sidebar.number_input("Kecepatan (km/jam)", 0, 150, 60)

    if st.sidebar.button("Prediksi Sekarang"):
        # Mapping hari ke angka (sesuai training)
        day_map = {"Senin":0, "Selasa":1, "Rabu":2, "Kamis":3, "Jumat":4, "Sabtu":5, "Minggu":6}
        
        # Encode detid
        det_enc = le.transform([detid])[0]
        
        # Buat array features (Urutan: detid_encoded, day_of_week, hour_of_day, occ, speed)
        features = np.array([[det_enc, day_map[day], hour, occ, speed]])
        
        # Prediksi
        prediction = model.predict(features)[0]
        
        # Logika Fuzzy Sederhana untuk Status
        if prediction < 200:
            status, color = "Lancar", "green"
        elif 200 <= prediction < 500:
            status, color = "Padat Merayap", "orange"
        else:
            status, color = "Macet Total", "red"
            
        # Tampilkan Hasil
        st.success(f"### Hasil Prediksi: {int(prediction)} Kendaraan")
        st.markdown(f"**Status Lalu Lintas:** :{color}[{status}]")