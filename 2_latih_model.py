import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# MENAMBAHKAN IMPORT METRICS BARU
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pickle

print("1. Membaca data dataset.csv...")
try:
    data = pd.read_csv('dataset.csv')
    # Ubah semua label jadi string biar aman (campuran angka & huruf)
    data['label'] = data['label'].astype(str) 
except FileNotFoundError:
    print("Dataset tidak ditemukan!")
    exit()

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Melatih Model (Huruf & Angka)...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("3. Melakukan Prediksi...")
y_pred = model.predict(X_test)

# --- BAGIAN PERHITUNGAN METRIK ---
akurasi = accuracy_score(y_test, y_pred)
# average='macro': Menghitung rata-rata tanpa mempedulikan ketimpangan jumlah data per kelas
presisi = precision_score(y_test, y_pred, average='macro', zero_division=0)
rekal = recall_score(y_test, y_pred, average='macro', zero_division=0)

print(f"\n=== HASIL EVALUASI ===")
print(f"Akurasi           : {akurasi * 100:.2f}%")
print(f"Presisi Rata-rata : {presisi * 100:.2f}%")
print(f"Rekal Rata-rata   : {rekal * 100:.2f}%")

# Opsional: Tampilkan laporan lengkap per huruf/angka jika ingin melihat detail
# print("\nDetail per Kelas:")
# print(classification_report(y_test, y_pred, zero_division=0))

with open('model_isyarat.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("\nModel berhasil disimpan ke 'model_isyarat.pkl'")