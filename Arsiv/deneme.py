from ultralytics import YOLO
import cv2
import time

# Modeli yükle
model = YOLO("Modeller/montaj_s_batuhan.pt")

# Video dosyasını aç
cap = cv2.VideoCapture(0)


# Video özelliklerini al
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # veya 'XVID'
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter ile çıktı videosunu oluştur
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# FPS hesaplama için değişkenler
prev_time = 0
curr_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS hesaplama
    curr_time = time.time()
    fps_calc = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # Her karede tahmin yap (sadece güven değeri ≥ 0.7 olanları tut)
    results = model(frame, conf=0.7)

    # Sonuçları çiz
    result_img = results[0].plot()

    # FPS değerini ekrana yaz
    cv2.putText(result_img, f"FPS: {fps_calc:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow("YOLO Video", result_img)

    # Sonucu dosyaya yaz
    out.write(result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()