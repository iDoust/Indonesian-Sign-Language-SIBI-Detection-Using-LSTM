# SIBI-Indonesian-Sign-Language-Detection
# Deteksi Bahasa Isyarat Indonesia (SIBI) Menggunakan LSTM

Proyek ini bertujuan untuk mengembangkan sistem pengenalan bahasa isyarat Indonesia (SIBI) berbasis LSTM (Long Short-Term Memory). Sistem ini mengumpulkan data gesture tangan, memprosesnya, melatih model, dan mengujinya untuk mengenali berbagai isyarat dari SIBI.

## 🔧 Fitur Utama
- Pengumpulan dataset gesture tangan secara real-time
- Pengecekan kelayakan data
- Pelatihan model LSTM dengan data gesture
- Pengujian model terhadap input gesture baru

## 📁 Struktur File
- CollectionLSTM.ipynb  # Gunakan kamera untuk mengumpulkan data gesture. Dataset akan disimpan dalam folder sesuai label.
- TestUsebleLSTM.ipynb  # Mengecek apakah data sudah sesuai format dan siap dilatih.
- TrainingLSTM.ipynb  # Melatih model LSTM menggunakan data yang telah dikumpulkan. Model akan disimpan dalam format .h5.
- TestingLSTM.ipynb  # Menguji akurasi model terhadap data baru secara real-time atau dari dataset pengujian.
- README.md  

## 🧠 Teknologi & Library
- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV
- Mediapipe
- Matplotlib

## 📊 Arsitektur Model
Model LSTM menggunakan:  
- Input Layer: Array urutan koordinat gesture (time-series)
- LSTM Layer: Untuk menangkap pola sekuensial dari gesture
- Dense Layer: Klasifikasi output ke label SIBI
- Softmax Output: Untuk menentukan gesture akhir

## 🎯 Tujuan Proyek
- Meningkatkan aksesibilitas bagi penyandang tunarungu
- Mendukung pengembangan sistem komunikasi berbasis AI
- Menggunakan teknologi Computer Vision & Deep Learning dalam dunia nyata

## 📌 Catatan
- Pastikan kamera berfungsi baik saat mengumpulkan data
- Gesture tangan harus konsisten dan sesuai label
- Semakin banyak data dikumpulkan, semakin baik performa model

## 📚 Referensi
- Mediapipe Hands
- TensorFlow LSTM Documentation
