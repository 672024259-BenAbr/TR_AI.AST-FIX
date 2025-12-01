# Deteksi Bahasa Isyarat (Sign Language Detection)

Program ini menggunakan **Python**, **OpenCV**, dan **MediaPipe** untuk mendeteksi gerakan tangan dan menerjemahkannya menjadi teks/kalimat secara real-time. Program menggunakan metode *Machine Learning* (Random Forest) berdasarkan koordinat kerangka tangan.

## 📋 1. Persyaratan Sistem (Wajib)

Agar library `mediapipe` berjalan lancar, Anda **harus** mengikuti aturan versi berikut:

  * **Sistem Operasi:** Windows / macOS / Linux (64-bit).
  * **Versi Python:** **Python 3.8 hingga Python 3.11**.
      * ⚠️ **PENTING:** Python 3.12, 3.13, atau 3.14 **BELUM DIDUKUNG** oleh MediaPipe saat ini.
      * Jika Anda memiliki Python versi baru, gunakan *Virtual Environment* (venv) dengan Python 3.11.
  * **Hardware:** Webcam (Laptop atau USB).

-----

## 🛠 2. Instalasi Library

Buka Terminal (VS Code / CMD) di folder proyek Anda, lalu jalankan perintah berikut:

### a. Aktifkan Virtual Environment (Jika ada)

Jika Anda menggunakan `.venv` (sangat disarankan):

```powershell
.\.venv\Scripts\Activate
```

*(Pastikan muncul tanda `(.venv)` di kiri terminal)*

### b. Instal Paket Python

Salin dan jalankan perintah ini untuk menginstal semua kebutuhan:

```powershell
pip install opencv-python mediapipe scikit-learn pandas numpy
```

**Daftar Library & Fungsinya:**

1.  `opencv-python`: Untuk akses kamera dan tampilan video.
2.  `mediapipe`: Untuk mendeteksi 21 titik koordinat tulang tangan.
3.  `scikit-learn`: Untuk algoritma kecerdasan buatan (Random Forest).
4.  `pandas`: Untuk membaca/menyimpan data CSV.
5.  `numpy`: Untuk operasi matematika matriks.

-----

## 🚀 3. Cara Menjalankan Program

Program harus dijalankan secara berurutan dari nomor 1 sampai 3.

### Langkah 1: Rekam Data (`1_rekam_data.py`)

Tujuannya untuk mengumpulkan contoh gerakan tangan Anda.

1.  Jalankan perintah:
    ```powershell
    python 1_rekam_data.py
    ```
2.  Jendela kamera akan terbuka. **KLIK KIRI satu kali pada jendela video** agar keyboard terbaca.
3.  **Cara Merekam:**
      * Arahkan tangan ke kamera sampai muncul garis-garis merah.
      * **Tahan tombol keyboard** sesuai huruf yang diinginkan.
      * Contoh: Buat pose "A", lalu tahan tombol **'A'** di keyboard.
      * Contoh Spasi: Buat pose khusus (misal jempol ke bawah), lalu tahan tombol **'SPASI'**.
4.  Lihat Terminal: Pastikan muncul pesan `✅ DATA DISIMPAN`.
5.  Kumpulkan minimal 50-100 data untuk setiap huruf/pose agar akurat.

### Langkah 2: Latih Model (`2_latih_model.py`)

Tujuannya untuk menyatukan data rekaman menjadi "Otak" AI.

1.  Pastikan file `dataset.csv` sudah ada (hasil dari langkah 1).
2.  Jalankan perintah:
    ```powershell
    python 2_latih_model.py
    ```
3.  Tunggu sampai muncul tulisan **"Akurasi: ...%"**.
4.  File baru bernama `model_isyarat.pkl` akan otomatis dibuat.

### Langkah 3: Jalankan Program Utama (`3_main_program.py`)

Ini adalah aplikasi penerjemah sesungguhnya.

1.  Jalankan perintah:
    ```powershell
    python 3_main_program.py
    ```
2.  **Cara Menggunakan:**
      * Bentuk isyarat tangan di depan kamera.
      * **Tahan posisi tangan** (diam) selama sekitar 0.5 detik (tunggu bar hijau penuh).
      * Huruf akan otomatis terketik di layar.
      * Gunakan pose "SPASI" untuk memisahkan kata.
3.  **Tombol Bantuan:**
      * Tekan **'C'**: Clear (Hapus semua teks).
      * Tekan **'Q'** atau klik **[X]**: Keluar dari program.

-----

## ❓ Pemecahan Masalah (Troubleshooting)

**1. Error: `ModuleNotFoundError: No module named 'mediapipe'`**

  * **Solusi:** Anda belum menginstal library atau salah lingkungan Python. Pastikan sudah mengetik `pip install mediapipe` di dalam lingkungan `(.venv)` yang aktif.

**2. Error: Program langsung keluar / Kamera tidak muncul**

  * **Solusi:** Index kamera mungkin salah. Buka file `1_rekam_data.py` atau `3_main_program.py`, cari baris `cap = cv2.VideoCapture(0)` lalu ubah angka `0` menjadi `1`.

**3. Tombol keyboard tidak berfungsi saat merekam**

  * **Solusi:** Fokus komputer Anda masih di Terminal. Arahkan mouse ke jendela video kamera, lalu **Klik Kiri** pada videonya. Baru tekan tombol huruf.

**4. Error saat instalasi pip: `No matching distribution found`**

  * **Solusi:** Anda menggunakan Python versi terlalu baru (3.12+). Uninstall Python tersebut dan install **Python 3.11 (64-bit)**.