import os
import cv2
import mediapipe as mp
import csv

# --- KONFIGURASI ---
FOLDER_DATASET = 'Dataset_Foto' 
FILE_CSV = 'dataset.csv'

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2, # Wajib 2 untuk SIBI
    min_detection_confidence=0.5
)

# 1. Buat Header CSV (42 titik x 3 = 126 kolom)
print(f"Sedang membuat file {FILE_CSV}...")
with open(FILE_CSV, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = ['label']
    # Tangan 1 (0-20) dan Tangan 2 (21-41)
    for i in range(42): 
        header += [f'x{i}', f'y{i}', f'z{i}']
    csv_writer.writerow(header)

# 2. Proses Folder
if not os.path.exists(FOLDER_DATASET):
    print(f"ERROR: Folder '{FOLDER_DATASET}' tidak ditemukan!")
    exit()

daftar_kelas = os.listdir(FOLDER_DATASET)
total_foto_berhasil = 0

for nama_kelas in daftar_kelas:
    path_kelas = os.path.join(FOLDER_DATASET, nama_kelas)
    if not os.path.isdir(path_kelas): continue
        
    print(f"📂 Memproses Kelas: {nama_kelas}")
    
    for nama_file in os.listdir(path_kelas):
        path_foto = os.path.join(path_kelas, nama_file)
        try:
            image = cv2.imread(path_foto)
            if image is None: continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # --- LOGIKA DATA 2 TANGAN ---
            if results.multi_hand_landmarks:
                data_baris = [nama_kelas] # Label
                
                data_tangan_total = [] 

                # Masukkan data tangan yang terdeteksi
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        data_tangan_total.extend([lm.x, lm.y, lm.z])
                
                # Padding: Jika cuma 1 tangan, sisanya isi 0 sampai total 126
                while len(data_tangan_total) < 126:
                    data_tangan_total.append(0.0)
                
                # Potong jika lebih (safety)
                data_tangan_total = data_tangan_total[:126]

                data_baris.extend(data_tangan_total)

                # Simpan ke CSV
                with open(FILE_CSV, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(data_baris)
                
                total_foto_berhasil += 1

        except Exception as e:
            print(f"Error pada {nama_file}: {e}")

print("-" * 40)
print(f"✅ Selesai. Total data: {total_foto_berhasil}")
print("Sekarang jalankan '2_latih_model.py'.")