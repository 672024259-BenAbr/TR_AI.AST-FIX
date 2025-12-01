import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

y_pred = model.predict(X_test)
print(f"3. Selesai! Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")

with open('model_isyarat.pkl', 'wb') as f:
    pickle.dump(model, f)