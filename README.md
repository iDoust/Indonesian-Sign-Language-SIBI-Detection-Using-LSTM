# 🤟 SIBI Detection Using LSTM

Sistem deteksi Bahasa Isyarat Indonesia (**SIBI**) secara real-time menggunakan arsitektur **LSTM** (Long Short-Term Memory) dan **MediaPipe Holistic**. Proyek ini mendeteksi 10 kelas gesture melalui input webcam dengan akurasi pengujian mencapai **~99.17%**.

## 📊 Hasil Evaluasi Model

| Metric | Training | Validation (Test) |
|---|---|---|
| **Accuracy** | 99.79% | **99.17%** |
| **Loss** | 0.0069 | **0.0856** |
| **Epochs** | 52 (Early Stopped) | - |

> [!NOTE]
> Model dilatih menggunakan 600 sampel (60 sekuens per kelas) dengan panjang 30 frame per sekuens. Setiap frame mengekstrak 258 keypoints (Pose & Tangan).

## 🛠️ Arsitektur Model
- **Input:** (30 frames, 258 keypoints)
- **LSTM Layers:** 3 layers (64 -> 128 -> 64 units) dengan ReLU activation.
- **Dropout:** 0.2 untuk mencegah overfitting.
- **Dense Layers:** 2 layers (64 -> 32) + 1 Output layer (Softmax).

## 📁 Struktur Proyek
```
📁 SIBI Detection Using LSTM/
├── 📁 Dataset/             # (Git-ignored) Koleksi data .npy
├── 📁 Model/               # Trained .h5 model
├── 📁 Notebooks/           # Notebook asli (untuk referensi)
│   ├── 1. CollectionLSTM.ipynb
│   ├── 2. TestUsebleLSTM.ipynb
│   ├── 3. TrainingLSTM.ipynb
│   └── 4. TestingLSTM.ipynb
├── 📁 Scripts/             # Skrip Python Modular (Refactor)
│   ├── utils.py            # Shared utilities
│   ├── 1. collect.py
│   ├── 2. validate.py
│   ├── 3. train.py
│   └── 4. predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Cara Penggunaan

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Koleksi Data (Opsional)
```bash
python Scripts/"1. collect.py" --actions "Nama" "Saya" --sequences 60
```

### 3. Validasi Data
```bash
python Scripts/"2. validate.py"
```

### 4. Training Model
```bash
python Scripts/"3. train.py" --epochs 100
```

### 5. Prediksi Real-time
```bash
python Scripts/"4. predict.py" --conf 0.75
```

## 🧪 Kelas Gesture (Actual)
`aku`, `dia`, `hai`, `kamu`, `maaf`, `nama`, `no_action`, `sehat`, `terima_kasih`, `tolong`.

---
*Dikembangkan menggunakan Python, TensorFlow, Keras, dan MediaPipe.*
