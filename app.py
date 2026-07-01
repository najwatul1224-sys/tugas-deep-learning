import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN UTAMA (AKADEMIS)
st.set_page_config(page_title="Prediksi Harga Pangan Strategis Kendari", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:28px; font-weight:bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size:18px; color: #4B5563; text-align: center; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 Aplikasi Prediksi Harga Pangan Strategis di Kota Kendari</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Menggunakan Multivariat GRU Berbasis Fitur Kedekatan Hari Raya</div>', unsafe_allow_html=True)

# List Hari Raya sesuai rancangan model
hari_raya_list = [
    '2020-05-24', '2021-05-13', '2022-05-02', '2023-04-22', '2024-04-10', '2025-03-31', '2026-03-20',
    '2020-12-25', '2021-12-25', '2022-12-25', '2023-12-25', '2024-12-25', '2025-12-25', '2026-12-25'
]
hari_raya_dt = pd.to_datetime(hari_raya_list)

def hitung_jarak_hari_raya(tgl):
    hari_raya_mendatang = hari_raya_dt[hari_raya_dt >= tgl]
    if len(hari_raya_mendatang) == 0:
        return 365
    return (hari_raya_mendatang.min() - tgl).days

# Fungsi Aktivasi Komputasi Matriks
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

# Forward Pass GRU Menggunakan Dictionary NumPy Weights (.npy)
def prediksi_gru_numpy(weights_path, X_input):
    w = np.load(weights_path, allow_pickle=True).item()
    
    # --- LAYER 1 GRU ---
    T, F = X_input.shape
    h_dim1 = w['gru1_recurrent'].shape[0]
    h1_seq = np.zeros((T, h_dim1))
    h_t = np.zeros(h_dim1)
    
    w_z, w_r, w_h = np.split(w['gru1_kernel'], 3, axis=1)
    u_z, u_r, u_h = np.split(w['gru1_recurrent'], 3, axis=1)
    
    bias1 = w['gru1_bias'].flatten()
    if len(bias1) == h_dim1 * 6:
        b_z_i = bias1[0 : h_dim1]
        b_r_i = bias1[h_dim1 : h_dim1*2]
        b_h_i = bias1[h_dim1*2 : h_dim1*3]
        b_z_r = bias1[h_dim1*3 : h_dim1*4]
        b_r_r = bias1[h_dim1*4 : h_dim1*5]
        b_h_r = bias1[h_dim1*5 : h_dim1*6]
    else:
        b_z_i = bias1[0 : h_dim1]
        b_r_i = bias1[h_dim1 : h_dim1*2]
        b_h_i = bias1[h_dim1*2 : h_dim1*3]
        b_z_r, b_r_r, b_h_r = np.zeros(h_dim1), np.zeros(h_dim1), np.zeros(h_dim1)

    for t in range(T):
        x_t = X_input[t]
        z_t = sigmoid(np.dot(x_t, w_z) + b_z_i + np.dot(h_t, u_z) + b_z_r)
        r_t = sigmoid(np.dot(x_t, w_r) + b_r_i + np.dot(h_t, u_r) + b_r_r)
        h_tilde = tanh(np.dot(x_t, w_h) + b_h_i + r_t * (np.dot(h_t, u_h) + b_h_r))
        h_t = (1 - z_t) * h_t + z_t * h_tilde
        h1_seq[t] = h_t

    # --- LAYER 2 GRU ---
    h_dim2 = w['gru2_recurrent'].shape[0]
    h_t2 = np.zeros(h_dim2)
    
    w_z2, w_r2, w_h2 = np.split(w['gru2_kernel'], 3, axis=1)
    u_z2, u_r2, u_h2 = np.split(w['gru2_recurrent'], 3, axis=1)
    
    bias2 = w['gru2_bias'].flatten()
    if len(bias2) == h_dim2 * 6:
        b_z2_i = bias2[0 : h_dim2]
        b_r2_i = bias2[h_dim2 : h_dim2*2]
        b_h2_i = bias2[h_dim2*2 : h_dim2*3]
        b_z2_r = bias2[h_dim2*3 : h_dim2*4]
        b_r2_r = bias2[h_dim2*4 : h_dim2*5]
        b_h2_r = bias2[h_dim2*5 : h_dim2*6]
    else:
        b_z2_i = bias2[0 : h_dim2]
        b_r2_i = bias2[h_dim2 : h_dim2*2]
        b_h2_i = bias2[h_dim2*2 : h_dim2*3]
        b_z2_r, b_r2_r, b_h2_r = np.zeros(h_dim2), np.zeros(h_dim2), np.zeros(h_dim2)

    for t in range(T):
        x_t2 = h1_seq[t]
        z_t2 = sigmoid(np.dot(x_t2, w_z2) + b_z2_i + np.dot(h_t2, u_z2) + b_z2_r)
        r_t2 = sigmoid(np.dot(x_t2, w_r2) + b_r2_i + np.dot(h_t2, u_r2) + b_r2_r)
        h_tilde2 = tanh(np.dot(x_t2, w_h2) + b_h2_i + r_t2 * (np.dot(h_t2, u_h2) + b_h2_r))
        h_t2 = (1 - z_t2) * h_t2 + z_t2 * h_tilde2

    # --- LAYER DENSE OUTPUT ---
    output = np.dot(h_t2, w['dense_kernel']) + w['dense_bias']
    return output[0]

# --- UI STREAMLIT SIDEBAR ---
st.sidebar.header("📁 Konfigurasi Data & Model")
uploaded_file = st.sidebar.file_uploader("Upload File CSV Harga Pangan Terbaru", type=["csv"])
model_dir = st.sidebar.text_input("Folder Penyimpanan Model", value="saved_models")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    
    if 'Komoditas_(Rp)' in df.columns:
        df = df.drop(columns=['Komoditas_(Rp)'])
        
    df['Jarak_Hari_Raya'] = df['Tanggal'].apply(hitung_jarak_hari_raya)
    df['Bulan'] = df['Tanggal'].dt.month
    
    daftar_komoditas = [col for col in df.columns if col not in ['Tanggal', 'Jarak_Hari_Raya', 'Bulan']]
    
    pilihan_komoditas = st.selectbox(
        "Pilih Komoditas Pangan Strategis Kendari:", 
        options=daftar_komoditas, 
        format_func=lambda x: x.replace('_', ' ')
    )
    
    weights_path = os.path.join(model_dir, f'weights_{pilihan_komoditas}.npy')
    scaler_path = os.path.join(model_dir, f'scaler_{pilihan_komoditas}.pkl')
    meta_path = os.path.join(model_dir, f'metadata_{pilihan_komoditas}.pkl')
    
    if os.path.exists(weights_path) and os.path.exists(scaler_path) and os.path.exists(meta_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            
        look_back = meta['look_back']
        fitur_kolom = meta['fitur_kolom']
        target_index = meta['target_index']
        
        matrix_data = df[fitur_kolom].values
        
        if st.button("🚀 Jalankan Forecast 30 Hari Ke Depan"):
            with st.spinner('Menghitung ramalan sekuensial via NumPy Matrix Engine...'):
                scaled_full = scaler.transform(matrix_data)
                future_sequence = scaled_full[-look_back:].copy()
                
                prediksi_30_hari = []
                list_tanggal_future = []
                tanggal_terakhir = df['Tanggal'].max()
                
                for i in range(30):
                    tgl_esok = tanggal_terakhir + timedelta(days=i+1)
                    list_tanggal_future.append(tgl_esok)
                    
                    pred_scaled = prediksi_gru_numpy(weights_path, future_sequence)
                    
                    dummy = np.zeros((1, len(fitur_kolom)))
                    dummy[0, target_index] = pred_scaled
                    pred_asli = scaler.inverse_transform(dummy)[0][target_index]
                    prediksi_30_hari.append(pred_asli)
                    
                    baris_baru = future_sequence[-1].copy()
                    baris_baru[target_index] = pred_scaled
                    
                    jarak_fitur_idx = fitur_kolom.index('Jarak_Hari_Raya')
                    baris_baru[jarak_fitur_idx] = (hitung_jarak_hari_raya(tgl_esok) - scaler.data_min_[jarak_fitur_idx]) / (scaler.data_max_[jarak_fitur_idx] - scaler.data_min_[jarak_fitur_idx])
                    
                    future_sequence = np.vstack([future_sequence[1:], baris_baru])
            
            # === PERBAIKAN & PENINGKATAN VISUAL UTAMA ===
            st.success(f"🎉 Komputasi Sukses! Model Multivariat GRU berhasil memproyeksikan data.")
            
            df_forecast = pd.DataFrame({'Tanggal': list_tanggal_future, 'Estimasi Harga (Rp)': prediksi_30_hari})
            df_forecast['Estimasi Harga (Rp)'] = df_forecast['Estimasi Harga (Rp)'].round(0)
            
            # --- 1. RINGKASAN METRIK STRATEGIS (STAT CARDS) ---
            st.markdown("### 📊 Ringkasan Indikator Ekonomi Pangan")
            harga_terakhir = df[pilihan_komoditas].iloc[-1]
            harga_pred_akhir = df_forecast['Estimasi Harga (Rp)'].iloc[-1]
            perubahan_persen = ((harga_pred_akhir - harga_terakhir) / harga_terakhir) * 100
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Harga Aktual Terakhir", value=f"Rp {harga_terakhir:,.0f}")
            with m2:
                st.metric(label="Prediksi Akhir Periode (30 Hari)", value=f"Rp {harga_pred_akhir:,.0f}")
            with m3:
                st.metric(
                    label="Estimasi Fluktuasi Harga", 
                    value=f"{perubahan_persen:+.2f}%",
                    delta=f"{harga_pred_akhir - harga_terakhir:+.0f} Rp",
                    delta_color="inverse"  # Merah saat harga naik, melambangkan inflasi pangan
                )
            
            # --- 2. ANALISIS KONTEKS FITUR HARI RAYA ---
            jarak_hari_raya_terakhir = hitung_jarak_hari_raya(tanggal_terakhir)
            st.info(
                f"📅 **Konteks Fitur Kedekatan:** Titik data historis terakhir berada pada tanggal **{tanggal_terakhir.strftime('%d-%m-%Y')}** "
                f"dengan estimasi jarak menuju Hari Raya keagamaan terdekat berikutnya adalah **{jarak_hari_raya_terakhir} Hari**. "
                f"Variabel ini secara multivariat memengaruhi sensitivitas tren proyeksi di bawah."
            )
            
            st.markdown("---")
            
            # --- 3. LAYOUT GRAFIK DAN TABEL DATA ---
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("📋 **Tabel Proyeksi Harga Harian**")
                st.dataframe(
                    df_forecast.style.format({'Tanggal': lambda t: t.strftime('%d-%m-%Y'), 'Estimasi Harga (Rp)': 'Rp {:,.0f}'}), 
                    height=450
                )
            with col2:
                st.write("📈 **Visualisasi Tren Pergerakan Harga**")
                fig = go.Figure()
                
                # Menampilkan 45 baris data historis agar tren transisi ke forecast terlihat luwes
                df_hist_last = df.tail(45)
                fig.add_trace(go.Scatter(
                    x=df_hist_last['Tanggal'], y=df_hist_last[pilihan_komoditas], 
                    name='Harga Aktual Historis', line=dict(color='#1E3A8A', width=2.5)
                ))
                fig.add_trace(go.Scatter(
                    x=df_forecast['Tanggal'], y=df_forecast['Estimasi Harga (Rp)'], 
                    name='Proyeksi Ke Depan (GRU)', line=dict(dash='dash', color='#EF4444', width=2.5)
                ))
                
                fig.update_layout(
                    xaxis_title="Dimensi Waktu (Tanggal)",
                    yaxis_title="Nilai Komoditas (Rupiah)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # --- 4. PENJELASAN METODOLOGI ILMIAH UNTUK DOSEN ---
            st.markdown("---")
            with st.expander("📚 Dokumen Informasi Metodologi Penelitian & Fitur Model"):
                st.markdown(
                    f"""
                    * **Ruang Lingkup Wilayah:** Pasar Tradisional di Kota Kendari, Sulawesi Tenggara.
                    * **Arsitektur Pemodelan:** Jaringan Saraf Tiruan Deep Learning berbasis *Stacked Gated Recurrent Unit* (GRU) dengan 2 Lapisan Tersembunyi (64 Units & 32 Units) dilengkapi regularisasi Dropout 20% guna memitigasi *overfitting*.
                    * **Sifat Multivariat:** Model tidak hanya membaca runtun waktu linier harga (*univariate*), melainkan mengintegrasikan matriks korelasi eksternal berupa **Fitur Kedekatan Hari Raya Besar** untuk menangkap anomali lonjakan permintaan pasar musiman.
                    * **Engine Eksekusi Aplikasi:** Forward-pass matematika murni dihitung secara lokal via NumPy Array Matrix Engine demi menjamin performa aplikasi yang ringan (*lightweight*) tanpa ketergantungan library TensorFlow pada komputer klien.
                    """
                )
    else:
        st.error(f"File bobot model (.npy) tidak ditemukan di folder '{model_dir}'.")
else:
    st.info("💡 Silakan unggah file CSV data pangan Kendari untuk memulai aplikasi.")
