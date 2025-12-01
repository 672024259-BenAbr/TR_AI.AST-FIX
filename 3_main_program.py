import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- KONFIGURASI ---
FILE_MODEL = 'model_isyarat.pkl'
NAMA_WINDOW = 'Aplikasi Bisindo (1 atau 2 Tangan)'
THRESHOLD_STABIL = 40 

# 1. Load Model
print("Memuat Model...")
try:
    with open(FILE_MODEL, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: File '{FILE_MODEL}' tidak ditemukan!")
    print("Harap jalankan program '2_latih_model.py' dulu.")
    exit()

# 2. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,              
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# 3. Buka Kamera
cap = cv2.VideoCapture(0)
# Setting resolusi standar
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow(NAMA_WINDOW)

kalimat = ""
prediksi_sebelumnya = ""
counter_stabil = 0

print("Kamera siap. Tunjukkan tangan Anda.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip & Warna
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # UI Header
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Kalimat: {kalimat}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    prediksi_saat_ini = ""

    if results.multi_hand_landmarks:
        # --- LOGIKA PENTING: KUMPULKAN SEMUA KOORDINAT ---
        data_input_ai = []
        
        # Loop semua tangan yang terlihat (bisa 1 atau 2)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for lm in hand_landmarks.landmark:
                data_input_ai.extend([lm.x, lm.y, lm.z])
        
        # --- SOLUSI UTAMA: PADDING DATA ---
        # Model butuh 126 data (42 titik x 3). 
        # Jika cuma 1 tangan (63 data), kita isi sisanya dengan 0.
        while len(data_input_ai) < 126:
            data_input_ai.append(0.0)
        
        # Potong jika kelebihan (jaga-jaga)
        data_input_ai = data_input_ai[:126]

        # --- PREDIKSI ---
        try:
            hasil = model.predict([data_input_ai])
            prediksi_saat_ini = str(hasil[0])
            
            # Tampilkan Huruf Melayang di Layar
            cv2.rectangle(frame, (250, 380), (390, 430), (255, 255, 255), -1)
            cv2.putText(frame, prediksi_saat_ini, (270, 420), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            
        except Exception as e:
            # Ini akan muncul di terminal jika ada error ukuran data
            print(f"Error Prediksi: {e}")

    # --- LOGIKA STABILISASI HURUF ---
    if prediksi_saat_ini != "":
        if prediksi_saat_ini == prediksi_sebelumnya:
            counter_stabil += 1
        else:
            counter_stabil = 0
            prediksi_sebelumnya = prediksi_saat_ini
        
        # Loading Bar Hijau
        if counter_stabil > 0:
            lebar = int((counter_stabil / THRESHOLD_STABIL) * 140)
            cv2.rectangle(frame, (250, 430), (250 + lebar, 440), (0, 255, 0), -1)

        # Jika sudah stabil, masukkan ke kalimat
        if counter_stabil == THRESHOLD_STABIL:
            if prediksi_saat_ini == "SPASI":
                kalimat += " "
            else:
                kalimat += prediksi_saat_ini
            counter_stabil = 0

    # Keyboard Control
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): break
    elif key & 0xFF == ord('c'): kalimat = "" # Clear
    elif key == 8: kalimat = kalimat[:-1]     # Backspace

    cv2.imshow(NAMA_WINDOW, frame)

cap.release()
cv2.destroyAllWindows()