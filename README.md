# Klasifikasi Gambar Rock Paper Scissors Menggunakan Convolutional Neural Network (CNN)

## Deskripsi Program
Program ini merupakan implementasi **Convolutional Neural Network (CNN)** menggunakan **TensorFlow** dan **Keras** untuk melakukan klasifikasi gambar permainan:

- Rock
- Paper
- Scissors

Model dilatih menggunakan dataset gambar yang disimpan dalam folder dataset dan mampu mengenali gambar berdasarkan kategori masing-masing.

Program melakukan proses:
- Membaca dataset gambar
- Melakukan preprocessing gambar
- Membagi data training dan validation
- Membuat model CNN
- Melatih model
- Mengevaluasi performa model
- Melakukan prediksi gambar

---

# Teknologi dan Library yang Digunakan

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- CNN (Convolutional Neural Network)

---

# Konsep yang Digunakan

Program menggunakan metode:

## Convolutional Neural Network (CNN)

CNN merupakan salah satu jenis Deep Learning yang sangat efektif untuk:
- Klasifikasi gambar
- Deteksi objek
- Computer vision
- Image recognition

CNN bekerja dengan cara:
- Mengekstraksi fitur gambar
- Mengenali pola visual
- Melakukan klasifikasi berdasarkan pola tersebut

---

# Dataset

Dataset yang digunakan adalah dataset gambar:

```bash
rockpaperscissors
```

Dataset terdiri dari 3 kelas:
- rock
- paper
- scissors

Struktur folder dataset:

```bash
rockpaperscissors/
│
├── rock/
├── paper/
└── scissors/
```

---

# Alur Program

## 1. Import Library

Program mengimpor library yang dibutuhkan untuk:
- Pengolahan data
- Deep learning
- Image preprocessing
- Pembuatan model CNN

```python
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

---

## 2. Menentukan Lokasi Dataset

```python
dataset_path = "./rockpaperscissors"
```

Variabel ini digunakan untuk menentukan lokasi folder dataset gambar.

---

## 3. Preprocessing Data Gambar

Program menggunakan `ImageDataGenerator` untuk:
- Normalisasi gambar
- Membagi data training dan validation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```

### Penjelasan
- `rescale=1./255`
  - Mengubah nilai pixel dari 0-255 menjadi 0-1
  - Membantu proses training lebih stabil

- `validation_split=0.2`
  - Membagi 20% data sebagai validation data

---

## 4. Membuat Data Training

```python
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
```

### Penjelasan Parameter
| Parameter | Fungsi |
|---|---|
| `target_size` | Mengubah ukuran gambar menjadi 150x150 |
| `batch_size` | Jumlah data tiap batch |
| `class_mode` | Menggunakan klasifikasi multi-kelas |
| `subset='training'` | Mengambil data training |

---

## 5. Membuat Data Validation

```python
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

Validation digunakan untuk:
- Mengukur performa model
- Menghindari overfitting

---

# Arsitektur Model CNN

Model CNN dibuat menggunakan `Sequential()`.

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(3, activation='softmax')
])
```

---

# Penjelasan Layer CNN

## 1. Conv2D Layer

```python
Conv2D(32, (3,3), activation='relu')
```

Fungsi:
- Mengekstraksi fitur gambar
- Mengenali pola seperti:
  - Garis
  - Bentuk
  - Tekstur

---

## 2. MaxPooling2D Layer

```python
MaxPooling2D(2,2)
```

Fungsi:
- Mengurangi ukuran feature map
- Mengurangi kompleksitas data
- Mempercepat training

---

## 3. Flatten Layer

```python
Flatten()
```

Fungsi:
- Mengubah data 2D menjadi 1D
- Agar dapat diproses oleh Dense layer

---

## 4. Dense Layer

```python
Dense(512, activation='relu')
```

Fungsi:
- Mempelajari pola hasil ekstraksi fitur

---

## 5. Output Layer

```python
Dense(3, activation='softmax')
```

Fungsi:
- Menghasilkan probabilitas 3 kelas:
  - Rock
  - Paper
  - Scissors

---

# Menampilkan Ringkasan Model

```python
model.summary()
```

Digunakan untuk melihat:
- Struktur model
- Jumlah parameter
- Bentuk input/output

---

# Compile Model

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

# Penjelasan Compile

| Parameter | Fungsi |
|---|---|
| `loss` | Mengukur kesalahan model |
| `optimizer='adam'` | Mengoptimalkan training |
| `metrics=['accuracy']` | Mengukur akurasi model |

---

# Training Model

```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)
```

### Penjelasan
- `epochs=10`
  - Model dilatih sebanyak 10 kali iterasi penuh dataset

- `validation_data`
  - Digunakan untuk evaluasi selama training

---

# Evaluasi Model

```python
val_loss, val_acc = model.evaluate(validation_generator)

print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')
```

Program menampilkan:
- Validation loss
- Validation accuracy

---

# Prediksi Model

```python
predictions = model.predict(validation_generator)

print(predictions)
```

Program melakukan prediksi terhadap data validation dan menghasilkan probabilitas tiap kelas.

---

# Cara Menjalankan Program

## 1. Install Dependency

```bash
pip install tensorflow numpy pandas
```

---

## 2. Siapkan Dataset

Pastikan struktur folder seperti berikut:

```bash
rockpaperscissors/
│
├── rock/
├── paper/
└── scissors/
```

---

## 3. Jalankan Program

```bash
python jst3.py
```

---

# Output Program

Program menghasilkan:
- Ringkasan model CNN
- Proses training model
- Nilai accuracy dan loss
- Hasil evaluasi validation
- Hasil prediksi gambar

---

# Contoh Output

```bash
Validation loss: 0.12
Validation accuracy: 0.96
```

Artinya:
- Model memiliki loss kecil
- Akurasi klasifikasi mencapai 96%

---
# Kesimpulan

Program ini berhasil mengimplementasikan Convolutional Neural Network (CNN) untuk klasifikasi gambar Rock Paper Scissors menggunakan TensorFlow dan Keras. Dengan memanfaatkan convolution layer dan pooling layer, model mampu mengenali pola gambar dan melakukan klasifikasi dengan cukup baik.

Program juga menerapkan preprocessing gambar dan validation data sehingga proses training menjadi lebih efektif dan terstruktur.

---
