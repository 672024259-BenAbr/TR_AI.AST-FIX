import cv2
import mediapipe as mp
import csv
import os

# --- KONFIGURASI ---
NAMA_FILE = 'dataset.csv'
NAMA_WINDOW = 'TEST KEYBOARD & HANDS'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Buat CSV
if not os.path.exists(NAMA_FILE):
    with open(NAMA_FILE, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header += [f'x{i}', f'y{i}', f'z{i}']
        csv_writer.writerow(header)

print("="*50)
print("   MODE DIAGNOSA (CEK KEYBOARD)")
print("="*50)
print("1. Klik Jendela Video agar aktif.")
print("2. Tekan tombol apa saja.")
print("3. Lihat terminal: Apakah muncul angka saat ditekan?")
print("="*50)

cap = cv2.VideoCapture(0)
cv2.namedWindow(NAMA_WINDOW)

total_simpan_sesi_ini = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Cek tombol Close
    try:
        if cv2.getWindowProperty(NAMA_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    except: pass

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    cv2.putText(frame, "Klik Jendela Video -> Tekan Tombol", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- BAGIAN PENTING: MENDETEKSI TOMBOL ---
    key = cv2.waitKey(1)
    
    # [DIAGNOSA] Jika tombol ditekan, beritahu di terminal
    if key != -1:
        print(f"DEBUG: Tombol ditekan! Kode Angka: {key}")
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hanya simpan jika tombol ditekan DAN ada tangan
            if key != -1:
                label_disimpan = ""
                if key == 32: label_disimpan = "SPASI"
                elif (48 <= key <= 57) or (97 <= key <= 122): label_disimpan = chr(key).upper()
                
                if label_disimpan != "":
                    data_tangan = [label_disimpan]
                    for lm in hand_landmarks.landmark:
                        data_tangan.extend([lm.x, lm.y, lm.z])
                    
                    with open(NAMA_FILE, mode='a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(data_tangan)
                    
                    total_simpan_sesi_ini += 1
                    cv2.putText(frame, f"TEREKAM: {label_disimpan}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    print(f"✅ DATA DISIMPAN: {label_disimpan}")

    cv2.imshow(NAMA_WINDOW, frame)

cap.release()
cv2.destroyAllWindows()