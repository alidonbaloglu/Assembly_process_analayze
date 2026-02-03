import cv2

# Kameraları aç (Windows için DirectShow backend ile)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Her iki kamera için çözünürlüğü 1000x800 olarak ayarla
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

if not cap.isOpened() or not cap1.isOpened():
    print("Kameralardan biri açılamadı.")
    if cap.isOpened():
        cap.release()
    if cap1.isOpened():
        cap1.release()
    cv2.destroyAllWindows()
    raise SystemExit

# İki ayrı pencere oluştur ve konumlandır
cv2.namedWindow("Kamera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Kamera1", cv2.WINDOW_NORMAL)
cv2.moveWindow("Kamera", 0, 0)
cv2.moveWindow("Kamera1", 700, 0)
cv2.resizeWindow("Kamera", 1000, 800)
cv2.resizeWindow("Kamera1", 1000, 800)

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()

    if not ret or not ret1:
        break

    # Kameradan alınan görüntüleri göster
    frame = cv2.resize(frame, (1000, 800))
    frame1 = cv2.resize(frame1, (1000, 800))
    cv2.imshow("Kamera", frame)
    cv2.imshow("Kamera1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap1.release()
cv2.destroyAllWindows()